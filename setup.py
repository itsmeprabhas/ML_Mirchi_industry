from setuptools import setup, find_packages

setup(
    name="ml-mirchi",
    version="1.0.0",
    description="Automated Chilli Pepper Quality Grading with Computer Vision",
    author="ML Mirchi Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "albumentations>=1.3.0",
        "opencv-python-headless>=4.8.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mirchi-train=train:main",
            "mirchi-eval=evaluate:main",
            "mirchi-export=export_model.py:main",
        ],
    },
)
