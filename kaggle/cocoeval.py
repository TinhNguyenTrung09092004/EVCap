"""COCO caption metrics for the json produced by eval_evcap.py.

Needs a JRE: PTBTokenizer, METEOR and SPICE are java jars inside pycocoevalcap.
`bash kaggle/setup.sh` installs one.
"""
import argparse
import json
import shutil

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_file_path', required=True)
    ap.add_argument('--spice', action='store_true', default=False)
    ap.add_argument('--out')
    args = ap.parse_args()

    if shutil.which('java') is None:
        raise SystemExit(
            "java not found: PTBTokenizer and METEOR cannot run, and scores from an "
            "untokenized comparison are not comparable to published numbers. "
            "Run `bash kaggle/setup.sh` first.")

    with open(args.result_file_path) as f:
        preds = json.load(f)
    print(f"{len(preds)} predictions from {args.result_file_path}")

    gts = {p['image_name']: [{'caption': c} for c in p['captions']] for p in preds}
    res = {p['image_name']: [{'caption': p['prediction']}] for p in preds}

    tokenizer = PTBTokenizer()
    gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)

    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
               (Meteor(), "METEOR"),
               (Rouge(), "ROUGE_L"),
               (Cider(), "CIDEr")]
    if args.spice:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(), "SPICE"))

    results = {}
    for scorer, name in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(name, list):
            for n, s in zip(name, score):
                results[n] = float(s)
        else:
            results[name] = float(score)

    print()
    for k, v in results.items():
        print(f"{k:<10} {v * 100:.1f}")

    out = args.out or args.result_file_path.replace('.json', '_metrics.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
