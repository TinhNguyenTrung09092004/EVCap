import os
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer

from dataset.coco_dataset import COCODataset, COCOKarpathyDataset
from models.evcap import EVCap
from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can drop the first `start_index` samples of its
    own shard, so a run killed by the Kaggle session limit resumes mid-epoch on
    exactly the batches it had not seen yet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_index:])

    def __len__(self):
        return max(self.num_samples - self.start_index, 0)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def trainable_state_dict(model):
    model_no_ddp = unwrap(model)
    keep = {k for k, v in model_no_ddp.named_parameters() if v.requires_grad}
    return {k: v for k, v in model_no_ddp.state_dict().items() if k in keep}


def save_checkpoint(model, optimizer, scaler, scheduler_state, path):
    save_obj = {
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        **scheduler_state,
    }
    tmp = path + ".tmp"
    torch.save(save_obj, tmp)
    os.replace(tmp, path)
    print(f"Saved checkpoint -> {path} (epoch {scheduler_state['epoch']}, step {scheduler_state['step']})")


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}") if args.distributed else torch.device(args.device)
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = args.accum_grad_iters
    output_dir = args.out_dir
    is_main = (not args.distributed) or get_rank() == 0
    if is_main and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    if args.distributed:
        sampler = ResumableDistributedSampler(
            dataset, shuffle=True, num_replicas=get_world_size(), rank=get_rank(), drop_last=True,
        )
        model = DistributedDataParallel(
            model, device_ids=[get_rank()], broadcast_buffers=False,
        )
    else:
        sampler = ResumableDistributedSampler(
            dataset, shuffle=True, num_replicas=1, rank=0, drop_last=True,
        )

    iters_per_epoch = len(sampler) // batch_size
    print(f"iters_per_epoch = {iters_per_epoch} (batch {batch_size} x world {get_world_size() if args.distributed else 1})")

    optimizer = set_optimizer(model, init_lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=epochs,
        iters_per_epoch=iters_per_epoch,
        min_lr=args.min_lr,
        init_lr=args.init_lr,
        decay_rate=None,
        warmup_start_lr=args.warmup_start_lr,
        warmup_steps=args.warmup_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    use_amp = scaler is not None
    print("use_amp", use_amp)

    start_epoch, start_step = 0, 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume {args.resume} does not exist")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if use_amp != (ckpt["scaler"] is not None):
            raise RuntimeError(
                f"checkpoint was written with amp={ckpt['scaler'] is not None} but this "
                f"run has amp={use_amp}; the optimizer state is not transferable")
        missing, unexpected = unwrap(model).load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has keys this model does not: {unexpected}")
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch, start_step = ckpt["epoch"], ckpt["step"]
        print(f"Resumed from {args.resume} at epoch {start_epoch} step {start_step}")

    model.train()
    last_path = os.path.join(output_dir, "last.pt")
    stopped_early = False

    for epoch in range(start_epoch, epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        sampler.set_epoch(epoch)
        offset = start_step if epoch == start_epoch else 0
        sampler.start_index = offset * batch_size

        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, sampler=sampler,
            shuffle=False, drop_last=True, num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        header = 'Train Epoch: [{}]'.format(epoch)

        step = offset
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader, args.print_freq, header)):
            step = offset + idx
            samples['image'] = samples['image'].to(device, non_blocking=True)
            scheduler.step(cur_epoch=epoch, cur_step=step)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(samples)["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss ({loss.item()}) at epoch {epoch} step {step}. "
                    f"With --lm_dtype fp32 this means fp16 autocast overflowed in the "
                    f"decoder; rerun with --no_amp to confirm.")

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if args.save_every > 0 and (step + 1) % args.save_every == 0 and is_main:
                save_checkpoint(model, optimizer, scaler,
                                {"epoch": epoch, "step": step + 1}, last_path)
            if args.max_steps > 0 and step + 1 >= args.max_steps:
                print(f"Reached --max_steps {args.max_steps}, stopping.")
                stopped_early = True
                break

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())

        if is_main:
            if stopped_early:
                # Mid-epoch stop: record where we actually are, so --resume takes
                # the rest of this epoch instead of skipping to the next one.
                save_checkpoint(model, optimizer, scaler,
                                {"epoch": epoch, "step": step + 1}, last_path)
            else:
                save_checkpoint(model, optimizer, scaler,
                                {"epoch": epoch + 1, "step": 0}, last_path)
                save_checkpoint(model, optimizer, scaler,
                                {"epoch": epoch + 1, "step": 0},
                                os.path.join(output_dir, f"{epoch:03d}.pt"))
        if stopped_early:
            break
    return model


def build_dataset(args):
    if args.dataset == "karpathy":
        return COCOKarpathyDataset(
            karpathy_json=args.karpathy_json,
            image_dirs={"train2014": args.train_dir, "val2014": args.val_dir},
            splits=tuple(args.karpathy_splits.split(",")),
            max_samples=args.max_samples,
        )
    return COCODataset(data_root=args.data_root)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--accum_grad_iters', type=int, default=1)
    parser.add_argument('--device', default='cuda', help='gpu for training')
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--amp', default=True)
    parser.add_argument('--no_amp', dest='amp', action='store_false')
    parser.add_argument('--dist_url', default="env://")
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--print_freq', type=int, default=50)

    parser.add_argument('--dataset', default='karpathy', choices=('karpathy', 'coco2014'))
    parser.add_argument('--data_root', default='data/coco/coco2014')
    parser.add_argument('--karpathy_json', default='/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json')
    parser.add_argument('--train_dir', default='/kaggle/input/datasets/nadaibrahim/coco2014/train2014/train2014')
    parser.add_argument('--val_dir', default='/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014')
    parser.add_argument('--karpathy_splits', default='train,restval')
    parser.add_argument('--max_samples', type=int, default=None)

    parser.add_argument('--lm', default='gpt2')
    parser.add_argument('--lm_dtype', default='fp32', choices=tuple(DTYPES))
    parser.add_argument('--vit_precision', default='fp16', choices=('fp16', 'fp32'))
    parser.add_argument('--retrieval_backend', default='torch', choices=('torch', 'faiss'))
    parser.add_argument('--ext_path', default='ext_data/ext_memory_lvis.pkl')
    parser.add_argument('--q_former_model',
                        default="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
    parser.add_argument('--max_txt_len', type=int, default=128)

    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=8e-5)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--resume', default='')
    parser.add_argument('--max_steps', type=int, default=-1)

    parser.add_argument('--num_query_token_txt', type=int, default=8)
    parser.add_argument('--topn', type=int, default=9)
    parser.add_argument('--disable_random_seed', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    if not args.disable_random_seed:
        set_seed(args.random_seed)
    init_distributed_mode(args)
    print(f'args: {vars(args)}')

    dataset = build_dataset(args)
    model = EVCap(
        ext_path=args.ext_path,
        vit_model="eva_clip_g",
        q_former_model=args.q_former_model,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision=args.vit_precision,
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=args.num_query_token_txt,
        topn=args.topn,
        llama_model=args.lm,
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=args.max_txt_len,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
        lm_dtype=DTYPES[args.lm_dtype],
        retrieval_backend=args.retrieval_backend,
    )
    train(dataset, model, args)


if __name__ == '__main__':
    main()
