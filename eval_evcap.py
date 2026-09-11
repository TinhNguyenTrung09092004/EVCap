import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from search import beam_search
import random
import numpy as np

from models.evcap import EVCap

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(img).unsqueeze(0)


def validation_coco_flickr30k(
    args,
    inpath,
    model,
    tokenizer,
) -> None:

    device = args.device
    with open(inpath, 'r') as infile:
        annotations = json.load(infile)
    items = list(annotations.items())
    if args.limit > 0:
        items = items[:args.limit]
    predicts = []
    for image_id, captions in tqdm(items):
        image_path = args.image_folder + image_id
        image = preprocess_image(image_path).to(device)
        with torch.amp.autocast("cuda", enabled=args.amp):
            qform_all_proj, atts_qform_all_proj = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list)
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1], device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(embeddings=embeddings, tokenizer=tokenizer,
                                   beam_width=args.beam_width, model=model.llama_model)
            sentence = sentence[0].split('#')[0].strip()

        if args.verbose:
            print('\n' + image_path)
            print('GT:   ', captions)
            print('Pred: ', sentence)

        predicts.append({
            "split": "valid",
            "image_name": image_id,
            "captions": captions,
            "prediction": sentence,
        })

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent=4)
    print(f"wrote {len(predicts)} predictions -> {out_json_path}")


@torch.no_grad()
def main(args) -> None:
    device = args.device
    print('load:', args.ckpt)
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
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    own = model.state_dict()
    bad = [k for k, v in new_state_dict.items() if k in own and own[k].shape != v.shape]
    if bad:
        raise RuntimeError(
            f"checkpoint/model shape mismatch on {bad} -- the checkpoint was trained "
            f"with a different decoder than --lm {args.lm}")
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"loaded {len(new_state_dict)} tensors from checkpoint, {len(unexpected)} unexpected")

    model.to(device)
    model.eval()
    validation_coco_flickr30k(args, args.path_of_val_datasets, model, model.llama_tokenizer)


if __name__ == '__main__':
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--name_of_datasets', default='coco', choices=('coco',))
    parser.add_argument('--path_of_val_datasets', default='./data/coco/test_captions.json')
    parser.add_argument('--image_folder', default='./data/coco/coco2014/val2014/')
    parser.add_argument('--out_path', default='./generated_captions.json')
    parser.add_argument('--ckpt', default='results/train_evcap/last.pt')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='amp', action='store_false')

    parser.add_argument('--lm', default='gpt2')
    parser.add_argument('--lm_dtype', default='fp32', choices=tuple(DTYPES))
    parser.add_argument('--vit_precision', default='fp16', choices=('fp16', 'fp32'))
    parser.add_argument('--retrieval_backend', default='torch', choices=('torch', 'faiss'))
    parser.add_argument('--ext_path', default='ext_data/ext_memory_lvis.pkl')
    parser.add_argument('--q_former_model',
                        default="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
    parser.add_argument('--max_txt_len', type=int, default=128)

    parser.add_argument('--num_query_token_txt', type=int, default=8)
    parser.add_argument('--topn', type=int, default=9)
    parser.add_argument('--beam_width', type=int, default=5, help='width of beam')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.random_seed)
    print('args: {}\n'.format(vars(args)))
    main(args)
