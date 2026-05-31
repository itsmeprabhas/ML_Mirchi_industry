# ML Mirchi  — Automated Chilli Quality Grading

![ML Mirchi Overview](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)

**ML Mirchi** is an end-to-end computer vision solution designed to automate the post-harvest quality grading of chillies. By replacing slow, subjective manual sorting with an AI-driven approach, this system provides consistent, millisecond-precision grading that can easily be integrated into real-time conveyor belt environments.

The project classifies chillies into four distinct categories:
- 🟢 **Grade A** (Premium Quality)
- 🟡 **Grade B** (Standard Quality)
- 🟠 **Grade C** (Substandard Quality)
- 🔴 **Reject** (Defective/Rotten)

---

##  Key Features

- **High-Speed Inference:** Uses **EfficientNet-B0** exported to **ONNX**, enabling robust, sub-30ms inference times on CPU, essential for real-time sorting.
- **Robust Training Pipeline:** Implements advanced augmentation techniques (via Albumentations), including simulated camera blur, lighting changes, CutMix, and MixUp to handle real-world deployment variations.
- **Interactive Dashboard UI:** Includes a beautiful, responsive web interface built with vanilla HTML/CSS/JS that allows users to upload images and view real-time grading metrics.
- **Production-Ready API:** A lightweight **FastAPI** backend that receives images and serves model predictions instantly.

---

##  Tech Stack

- **Deep Learning:** PyTorch, Torchvision, Albumentations
- **Production & Inference:** ONNX, ONNXRuntime
- **Backend API:** FastAPI, Uvicorn
- **Frontend UI:** HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Data Analysis:** Scikit-Learn, Pandas, Matplotlib, Seaborn

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/itsmeprabhas/ML_Mirchi_industry.git
cd ML_Mirchi_industry

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Running the Application
The application consists of a FastAPI backend and a web frontend.

**Start the Backend Server:**
```bash
# This runs the API on http://localhost:8000
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Open the Frontend:**
Simply open `index.html` in your favorite web browser. You can now drag and drop chilli images into the grading lab to see the model in action!

---

##  Model Pipeline (For Developers)

If you want to train the model from scratch, evaluate it, or export a new version, follow these steps.

### 1. Training the Model
Ensure your data is organized inside the `data/processed/` directory with `train/` and `val/` subfolders containing class directories (`Grade_A`, `Grade_B`, `Grade_C`, `Reject`).

```bash
python train.py --model efficientnet_b0 --epochs 30 --batch-size 32
```
This will train the model and save the best weights to the `checkpoints/` directory.

### 2. Evaluating the Model
To generate performance metrics (Accuracy, Precision, Recall, F1) and visualize the confusion matrix and latency distributions:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```
Results, including metric reports and plots, will be saved to the `results/` folder.

### 3. Exporting for Production
To achieve the lowest latency possible, export the PyTorch `.pth` checkpoint to an ONNX graph:

```bash
python export_model.py --format onnx
```
The optimized model will be saved to the `exports/` directory as `mirchi_grader.onnx`.

---

##  Project Structure

```
ML_Mirchi/
│
├── app.py                  # FastAPI backend server
├── train.py                # Model training script
├── evaluate.py             # Evaluation & metric generation script
├── export_model.py         # Script to export PyTorch models to ONNX
│
├── index.html              # Main web application UI
├── static/                 # Frontend assets (CSS & JS)
│   ├── css/style.css
│   └── js/app.js
│
├── checkpoints/            # Saved PyTorch model weights (.pth)
├── exports/                # Optimized production models (.onnx)
├── results/                # Evaluation plots and metric JSONs
├── data/                   # Dataset folder (raw and processed)
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

##  License
This project is for educational and industrial demonstration purposes.
