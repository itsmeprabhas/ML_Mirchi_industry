# ML Mirchi - Automated Chilli Quality Grading

A computer vision project that classifies chillies into quality grades (Grade A, Grade B, Grade C, Reject) using an EfficientNet-B0 model trained in PyTorch and deployed with ONNX and FastAPI.

## Project Structure
- `train.py`: Model training script with Albumentations, MixUp, and CutMix.
- `evaluate.py`: Generates evaluation metrics, confusion matrix, and latency benchmarks.
- `export_model.py`: Exports trained model to ONNX for fast inference.
- `app.py`: FastAPI backend that loads the ONNX model and serves predictions.
- `index.html`, `static/css`, `static/js`: The web frontend.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API backend:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. **Open the frontend:**
   Simply open `index.html` in your web browser.

## Training
To train the model from scratch, place images in `data/processed/train` and `data/processed/val`, then run:
```bash
python train.py --epochs 30 --batch-size 32
```
