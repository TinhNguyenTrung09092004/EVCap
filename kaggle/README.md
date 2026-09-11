# Chạy EVCap với decoder GPT-2 Small trên Kaggle
`transformers 5.0.0`, `torch 2.10.0+cu128`, GPU T4 x2.

## 1. Bootstrap

```
!git clone -b kaggle https://github.com/TinhNguyenTrung09092004/EVCap.git /kaggle/working/EVCap
%cd /kaggle/working/EVCap
```

## 2. Các bước

```bash
!bash kaggle/setup.sh
!python kaggle/prepare_data.py
!python kaggle/smoke_test.py --full
!bash kaggle/run_train.sh
!bash kaggle/run_eval.sh
```

## 3. Resume

```bash
!bash kaggle/run_train.sh --resume results/gpt2/last.pt
```
