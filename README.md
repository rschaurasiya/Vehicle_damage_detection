# Vehicle Damage Detection

Vehicle Damage Detection is a Streamlit application that uses a fine-tuned
ResNet-50 model to classify an uploaded vehicle image. The model predicts
whether the image shows the front or rear of a vehicle and whether that view
is normal, crushed, or broken.

## Application Screenshot

![Car Damage Detection Streamlit application](Screenshot%20%28666%29.png)

The screenshot shows the application after uploading a vehicle image and receiving a rear-normal (`R_Normal`) prediction.
## Try the application

- **Live Streamlit app:** [vehicledamagedetection-byradheshyam.streamlit.app](https://vehicledamagedetection-byradheshyam.streamlit.app/)
- **GitHub repository:** [rschaurasiya/Vehicle_damage_detection](https://github.com/rschaurasiya/Vehicle_damage_detection)

Upload a `.jpg` or `.png` image in the application to receive a prediction.

## Features

- Upload a vehicle image through a browser.
- Classify the image into one of six vehicle-condition classes.
- Display the uploaded image and the predicted class.
- Run inference locally with the included PyTorch model.
- Review the training notebooks and dataset layout used for the classifier.

## Prediction classes

| Class | Description |
| --- | --- |
| `Front Breakage` | Front view with breakage |
| `Front Crushed` | Front view with crushed damage |
| `Front Normal` | Normal front view |
| `Rear Breakage` | Rear view with breakage |
| `Rear Crushed` | Rear view with crushed damage |
| `Rear Normal` | Normal rear view |

## How the application works

1. The user uploads a JPG or PNG image.
2. The image is converted to RGB, resized to `224 x 224`, and normalized
   using ImageNet statistics.
3. The fine-tuned ResNet-50 checkpoint is loaded from
   `streamlit_app/model/saved_model.pth`.
4. The model predicts one of the six classes above and displays the result.

## Installation

Follow these steps to run the application on your own computer.

### 1. Install the prerequisites

- Python 3.10 or newer
- Git
- Internet access for installing Python packages

PyTorch runs on the CPU by default in this project, so a GPU is not required.

### 2. Clone the repository

```bash
git clone https://github.com/rschaurasiya/Vehicle_damage_detection.git
cd Vehicle_damage_detection
```

### 3. Create and activate a virtual environment

Creating a virtual environment keeps this project's packages separate from
other Python projects.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, open PowerShell as your user and run
the following command once, then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the dependencies

Run this command from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r streamlit_app/requirements.txt
```

The requirements file installs Streamlit, PyTorch, torchvision, and Pillow.
The repository already contains the trained model, so no separate model
download is required.

### 5. Start the application

From the repository root, run:

```bash
python -m streamlit run streamlit_app/app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`, and upload a JPG or PNG vehicle image.

To stop the application, press `Ctrl+C` in the terminal. To leave the virtual
environment, run `deactivate`.

## Project structure

```text
Vehicle_damage_detection/
├── README.md
├── Screenshot (666).png
├── streamlit_app/
│   ├── app.py                      # Streamlit user interface
│   ├── model_helper.py             # Model definition and prediction logic
│   ├── model/
│   │   └── saved_model.pth         # Trained model checkpoint
│   └── requirements.txt            # Application dependencies
├── FastApi/
│   ├── server.py                   # Optional FastAPI endpoint
│   └── requirements.txt            # API dependencies
└── training/
    ├── dataset/                    # ImageFolder-compatible dataset
    ├── project_car_damage_1.ipynb
    └── hyperparameter_tunning_with_Resnet50.ipynb
```

## Training data layout

The training notebooks use the `torchvision.datasets.ImageFolder` format. Each
class must have its own directory:

```text
dataset/
├── Front Breakage/
├── Front Crushed/
├── Front Normal/
├── Rear Breakage/
├── Rear Crushed/
└── Rear Normal/
```

The model was trained with a ResNet-50 backbone, `224 x 224` input images, and
six output classes. If you replace the checkpoint, its architecture and class
ordering must remain compatible with `streamlit_app/model_helper.py`.

## Optional FastAPI service

The `FastApi/` directory contains a separate API implementation with a
`POST /predict` endpoint. The Streamlit application is the recommended and
documented way to use this project. If you use the API, install its
dependencies from that directory and run it with your preferred ASGI server.

## Troubleshooting

- **`python` is not recognized:** install Python, restart the terminal, and
  ensure Python is available on your PATH. On some systems, use `python3`
  instead.
- **The model file cannot be found:** run the command from the cloned
  repository and confirm that
  `streamlit_app/model/saved_model.pth` exists.
- **Package installation fails:** confirm that the virtual environment is
  active and use a Python version supported by the dependency versions in
  `streamlit_app/requirements.txt`. For platform-specific PyTorch wheels,
  consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
- **The upload is rejected:** the current interface accepts `.jpg` and `.png`
  files only.

## Limitations and responsible use

This project is an image-classification demonstration and is not a substitute
for a professional vehicle inspection, repair estimate, insurance decision, or
legal assessment. Prediction quality can vary with image quality, lighting,
viewpoint, and how closely an image matches the training data.

