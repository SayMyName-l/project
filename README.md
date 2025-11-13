# project

Run training directly in a Kaggle Notebook with the dataset mounted at `Input/datass`:

```bash
!python main.py \
    --dataset dogs \
    --data_root /kaggle/input/datass \
    --epochs 5 \
    --batch_size 64 \
    --val_ratio 0.1
```

The script automatically splits `/train` into training/validation folds when a dedicated
`val` folder is absent and reads the competition `test1` directory for submission
generation. Predictions are saved to `outputs/submission.csv`.
