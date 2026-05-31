# ML Mirchi — Retraining Guide

## Schedule
- **Frequency**: Monthly (or when drift detected)
- **Data**: Flagged samples + new production data
- **Validation**: Always validate on held-out test set before deployment

## Steps

```bash
# 1. Add new images to data/raw/
# 2. Re-run dataset preparation
python data/prepare_dataset.py

# 3. Retrain (resume from best checkpoint)
python train.py --epochs 10

# 4. Evaluate on test set
python evaluate.py

# 5. Export if metrics improved
python export_model.py --format onnx

# 6. Deploy new model
```
