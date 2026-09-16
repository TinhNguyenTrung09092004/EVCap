# Running EVCap with a GPT-2 Small decoder on Kaggle

## 1. Notebook cells

### Cell 1

```
!git clone -b kaggle https://github.com/TinhNguyenTrung09092004/EVCap.git /kaggle/working/EVCap
%cd /kaggle/working/EVCap
!bash kaggle/setup.sh
!python kaggle/prepare_data.py
```

### Cell 2

```bash
!bash kaggle/run_train.sh --random_seed 0
!bash kaggle/run_eval.sh  --random_seed 0
```

## 2. Resume

```bash
!bash kaggle/run_train.sh --resume results/gpt2/last.pt
```
