"""Fail fast on the Kaggle image: report versions, then run one real train step.

    python kaggle/smoke_test.py            # imports + versions only
    python kaggle/smoke_test.py --full     # + build EVCap and do a forward/backward
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def versions():
    import torch, transformers, timm, torchvision
    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"torchvision  {torchvision.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"timm         {timm.__version__}")
    print(f"cuda         {torch.version.cuda} | devices {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        print("  note: pre-Ampere GPU -> bf16 unavailable, use fp16 AMP")


def imports():
    import models.Qformer, models.Qformer_txt, models.eva_vit, models.blip2  # noqa
    from models.evcap import EVCap  # noqa
    from search import beam_search  # noqa
    from dataset.coco_dataset import COCOKarpathyDataset  # noqa
    print("repo imports OK")


def full(args):
    import torch
    from models.evcap import EVCap

    model = EVCap(
        ext_path=args.ext_path,
        q_former_model=args.q_former_model,
        num_query_token_txt=8,
        topn=9,
        llama_model=args.lm,
        max_txt_len=128,
        lm_dtype=torch.float32,
        retrieval_backend=args.retrieval_backend,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\ntrainable params {n_train:,} / {n_all:,}")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:<40} {tuple(p.shape)}")

    device = "cuda:0"
    model.to(device).train()
    samples = {
        "image": torch.randn(2, 3, 224, 224, device=device),
        "text_input": ["a man riding a horse on a beach", "two dogs playing in the snow"],
    }
    trainable = [p for p in model.parameters() if p.requires_grad]
    with torch.amp.autocast("cuda", enabled=True):
        out = model(samples)
    loss = out["loss"]
    print(f"\nloss = {loss.item():.4f}  (finite: {torch.isfinite(loss).item()})")
    assert torch.isfinite(loss), "non-finite loss"

    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    grads = [(n, p.grad.abs().mean().item()) for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
    print("unscaled grad means:", {n: f"{g:.3e}" for n, g in grads})
    bad = [n for n, g in grads if g != g or g == float("inf")]
    assert not bad, f"non-finite unscaled grads: {bad}"

    opt = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    model.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scale0 = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    skipped = scaler.get_scale() < scale0
    print(f"GradScaler first step: init_scale {scale0:g} -> {scaler.get_scale():g} "
          f"({'overflowed, step skipped (normal warm-up)' if skipped else 'applied'})")
    model.zero_grad(set_to_none=True)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    model.eval()
    from search import beam_search
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        qp, ap = model.encode_img(samples["image"][:1])
        pe, _ = model.prompt_wrap(qp, ap, model.prompt_list)
        bos = torch.full((1, 1), model.llama_tokenizer.bos_token_id, device=device, dtype=torch.long)
        emb = torch.cat([model.embed_tokens(bos), pe], dim=1)
        sent = beam_search(embeddings=emb, tokenizer=model.llama_tokenizer,
                           beam_width=3, model=model.llama_model)
    print(f"\nuntrained sample caption: {sent[0]!r}")
    print("\nSMOKE TEST PASSED")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--lm', default='gpt2')
    ap.add_argument('--ext_path', default='ext_data/ext_memory_lvis.pkl')
    ap.add_argument('--retrieval_backend', default='torch')
    ap.add_argument('--q_former_model',
                    default="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
    a = ap.parse_args()
    versions()
    print()
    imports()
    if a.full:
        full(a)
