"""Build the eval annotation file EVCap expects, and sanity-check the Kaggle mounts.

Writes `<out_dir>/test_captions.json` as {image_filename: [caption, ...]} over the
Karpathy COCO test split, which `eval_evcap.py` reads with
`image_path = args.image_folder + image_id`.
"""
import argparse
import json
import os
from collections import Counter

KARPATHY = '/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json'
TRAIN_DIR = '/kaggle/input/datasets/nadaibrahim/coco2014/train2014/train2014'
VAL_DIR = '/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014'


def check_dirs(dirs):
    for name, path in dirs.items():
        if not os.path.isdir(path):
            raise SystemExit(f"[FATAL] {name} directory not found: {path}\n"
                             f"        Attach the dataset or pass --{name.replace('2014', '')}_dir")
        n = sum(1 for _ in os.scandir(path))
        print(f"  {name:<10} {path}  ({n} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--karpathy_json', default=KARPATHY)
    ap.add_argument('--train_dir', default=TRAIN_DIR)
    ap.add_argument('--val_dir', default=VAL_DIR)
    ap.add_argument('--out_dir', default='data/coco')
    ap.add_argument('--splits', default='test,val')
    args = ap.parse_args()

    print("Checking image directories:")
    dirs = {'train2014': args.train_dir, 'val2014': args.val_dir}
    check_dirs(dirs)

    if not os.path.isfile(args.karpathy_json):
        raise SystemExit(f"[FATAL] Karpathy json not found: {args.karpathy_json}")
    with open(args.karpathy_json) as f:
        images = json.load(f)['images']
    print(f"\nKarpathy file: {len(images)} images")
    print("  split sizes:", dict(Counter(im['split'] for im in images)))

    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits.split(','):
        split = split.strip()
        subset = [im for im in images if im['split'] == split]
        if not subset:
            print(f"  [skip] no images in split {split!r}")
            continue

        folders = set(im['filepath'] for im in subset)
        if len(folders) > 1:
            raise SystemExit(
                f"[FATAL] split {split!r} spans {folders}; eval_evcap.py prefixes every key "
                f"with a single --image_folder, so this needs a code change.")
        folder = folders.pop()

        ann = {}
        missing = 0
        for im in subset:
            path = os.path.join(dirs[folder], im['filename'])
            if not os.path.exists(path):
                missing += 1
                if missing <= 3:
                    print(f"  [warn] missing image {path}")
                continue
            ann[im['filename']] = [s['raw'].strip() for s in im['sentences']]

        out = os.path.join(args.out_dir, f'{split}_captions.json')
        with open(out, 'w') as f:
            json.dump(ann, f)
        n_caps = sum(len(v) for v in ann.values())
        print(f"  wrote {out}: {len(ann)} images / {n_caps} captions "
              f"(folder={folder}, missing={missing})")
        print(f"    -> use --image_folder {dirs[folder]}/")

    train = [im for im in images if im['split'] in ('train', 'restval')]
    print(f"\nKarpathy train+restval: {len(train)} images / "
          f"{sum(len(im['sentences']) for im in train)} captions")


if __name__ == '__main__':
    main()
