# ASL Alphabet Classification Project

This project implements a Convolutional Neural Network (CNN) to classify American Sign Language (ASL) alphabets from images. It includes a training pipeline, a real-time detection GUI, and tools to export the model for edge devices (Edge Impulse/TFLite).

## Project Structure

```text
ASL-SBS-SES/
│
├── dataset/                  # Run download_dataset.bash download the dataset
├── models/                   # Saved .keras and .tflite models
├── src/                      # Source code
│   ├── config.py             # Configuration (Paths, Hyperparams)
│   ├── data.py               # Data loading & Augmentation
│   ├── model.py              # CNN Architecture
│   └── utils.py              # GPU setup & Helper functions
│
├── train.py                  # Script to train the model
├── app.py                    # Real-time Webcam GUI
├── quantize.py               # Quantization script for TFLite/Edge Impulse
├── predict.py                # Single image inference script
├── convert_to_tflite.py      # Script to convert the .keras model to .tflite
└── README.md
```
## Setup & Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for fast and efficient dependency management.
The dataset that was used is from Kaggle [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)

### Initialize Environment

Clone and download the dataset:

```bash
# Clone the repo
git clone https://github.com/S1abu/ASL-SBS-SES && cd ASL-SBS-SES/

# Download dependencies
uv sync

# Download the dataset
bash download_dataset.bash
```

*(Note: To ensure GPU support, make sure you have the appropriate NVIDIA drivers and CUDA toolkit installed for your system).*

---

## Training the Model

To train the CNN model from scratch:

1. **Prepare Data:** Ensure your dataset is located at `dataset/asl_alphabet_train/asl_alphabet_train`.
2. **Run Training:**
```bash
uv run train.py
```

3. **Output:**
* The best model is saved to `models/asl_model.keras`.
* Training accuracy and loss graphs will verify performance.

*You can modify batch size, epochs, or learning rate in `src/config.py`.*

---

## Real-Time Detection App

Run the live webcam interface to test the model:

```bash
uv run app.py
```

**How to use:**

1. A window will open showing your webcam feed.
2. Locate the **Green Box** (Region of Interest) on the screen.
3. Place your hand **inside the green box** to get a prediction.
* *Tip:* Keep the background inside the box as neutral as possible.



---

## Deployment (Edge Impulse / TFLite)

To export the model for edge devices (microcontrollers, mobile) or Edge Impulse:

### 1. Quantize the Model

Run the quantization script to convert the Keras model to a TFLite Int8 model:

```bash
uv run quantize.py
```

This generates `models/model_quantized.tflite`.

### 2. Upload to Edge Impulse

1. Go to the **Upload** section in your Edge Impulse project.
2. Select **Upload your model**.
3. Choose the `model_quantized.tflite` file.
4. **Important:** Select **"Image"** as the input type.

---

## Configuration

You can easily adjust project settings in `src/config.py` without changing the core code:

```python
# src/config.py snippet
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 128   # Adjust based on your GPU VRAM
EPOCHS = 20

```

## Troubleshooting

* **GPU not detected?**
The `train.py` script attempts to enable memory growth. If it fails, check that `nvidia-smi` works in your terminal and that you have installed `tensorflow` (Linux/WSL) or `tensorflow-cpu` + `tensorflow-directml` (if using specific setups), though standard `pip install tensorflow` usually covers CUDA on Linux/WSL.
* **"Model not found" error?**
Ensure you run `uv run train.py` successfully at least once before running `app.py` or `quantize.py`.

```

```