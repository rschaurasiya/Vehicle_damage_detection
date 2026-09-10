# Car Damage Detection

An image-classification application that uses a fine-tuned ResNet-50 model to identify the condition and viewpoint of a vehicle in an uploaded image. The included Streamlit interface provides a simple way to run predictions locally.

## Application Screenshot

![Car Damage Detection Streamlit application](Screenshot%20%28666%29.png)

The screenshot shows the application after uploading a vehicle image and receiving a rear-normal (`R_Normal`) prediction.

## Features

- Upload vehicle images in `.jpg` or `.png` format.
- Classify images into six vehicle-condition categories.
- Display the uploaded image and predicted class in the browser.
- Run inference locally with a packaged PyTorch model.
- Explore the training and hyperparameter-tuning workflow in Jupyter notebooks.

## Prediction Classes

The model predicts one of the following classes:

| Class | Meaning |
| --- | --- |
| `F_Breakage` | Front view with breakage |
| `F_Crushed` | Front view with crushed damage |
| `F_Normal` | Normal front view |
| `R_Breakage` | Rear view with breakage |
| `R_Crushed` | Rear view with crushed damage |
| `R_Normal` | Normal rear view |

## How It Works

1. The user uploads a vehicle image through the Streamlit interface.
2. The image is converted to RGB, resized to `224 x 224`, converted to a tensor, and normalized using ImageNet statistics.
3. A fine-tuned ResNet-50 model loads the saved weights and performs inference on the CPU.
4. The predicted class is displayed in the application.

## Project Structure

```text
Car_Damage_Detection/
├── README.md
├── saved_model.pth                 # Root-level model checkpoint
├── streamlit_app/
│   ├── app.py                      # Streamlit user interface
│   ├── model_helper.py             # Model definition and prediction logic
│   ├── model/
│   │   └── saved_model.pth         # Checkpoint used by the app
│   └── requirements.txt
└── training/
    ├── dataset/                    # ImageFolder-compatible training data
    ├── project_car_damage_1.ipynb  # Initial training workflow
    └── hyperparameter_tunning_with_Resnet50.ipynb
```

## Installation

### Prerequisites

- Python 3.10 or later
- A virtual environment is recommended

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the application dependencies:

```bash
pip install -r streamlit_app/requirements.txt
```

> PyTorch and torchvision wheels can vary by operating system and CUDA version. If dependency installation fails, install compatible versions for your platform from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining requirements.

## Run the Application

From the repository root, start Streamlit from the application directory:

```bash
cd streamlit_app
streamlit run app.py
```

Streamlit will print a local URL, typically `http://localhost:8501`. Open it in a browser, upload a supported image, and review the predicted class.

## Training

The `training/` directory contains the notebooks used to prepare the dataset and train the classifier. The dataset follows the `torchvision.datasets.ImageFolder` layout, with one subdirectory per class:

```text
dataset/
├── F_Breakage/
├── F_Crushed/
├── F_Normal/
├── R_Breakage/
├── R_Crushed/
└── R_Normal/
```

The documented workflow uses:

- ResNet-50 with ImageNet pretrained weights
- `224 x 224` input images
- Image augmentation during training
- A 75/25 train-validation split
- A six-class output layer

To use a newly trained checkpoint in the application, place it at `streamlit_app/model/saved_model.pth` and ensure its architecture and class ordering match the inference code.

## Limitations

- Predictions are limited to the six classes listed above.
- The model is intended as an assistive classification tool, not a substitute for professional vehicle inspection or insurance assessment.
- Prediction quality depends on image quality, lighting, viewpoint, and similarity to the training data.
- The application currently accepts JPG and PNG uploads.

## Responsible Use

Validate predictions against representative held-out data before using the model in a production workflow. Avoid using a single prediction as the sole basis for safety, repair, insurance, or legal decisions.

