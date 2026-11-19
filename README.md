# Digit Recognition Web Application

A handwritten digit recognition web application built using **Python, TensorFlow, Keras, and Flask**. The application uses a trained deep learning model to recognize handwritten digits (0–9) from uploaded images and displays the predicted digit along with its confidence score.

## Features

- Recognizes handwritten digits from 0 to 9.
- Uses a Deep Learning neural network built with TensorFlow and Keras.
- Upload images through a simple Flask web interface.
- Converts uploaded images into grayscale.
- Resizes images to 28 × 28 pixels.
- Normalizes pixel values before prediction.
- Displays the predicted digit and confidence percentage.

## Technologies Used

- **Python** – Programming language
- **TensorFlow & Keras** – Deep Learning model
- **NumPy** – Numerical operations and image processing
- **Pillow (PIL)** – Image processing
- **Flask** – Web application framework
- **MNIST Dataset** – Handwritten digit dataset

## Project Structure

```text
digit-recognition/
│
├── app.py
├── train.py
├── digit_model.keras
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

## Deep Learning Model

The model is trained using the MNIST handwritten digit dataset.

### Model Architecture

1. Input Layer – Accepts 28 × 28 pixel images.
2. Flatten Layer – Converts the image into a one-dimensional array.
3. Dense Layer – 128 neurons with ReLU activation.
4. Output Layer – 10 neurons with Softmax activation for digit classification.

### Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 5
- Validation Split: 10%
- Dataset: MNIST

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digit-recognition.git
```

### 2. Navigate to the Project Folder

```bash
cd digit-recognition
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model (Optional)

If you want to train the model again:

```bash
python train.py
```

This generates the trained model file:

```text
digit_model.keras
```

### 6. Run the Flask Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## How It Works

1. The user uploads an image containing a handwritten digit.
2. Flask receives the uploaded image.
3. The image is converted into grayscale.
4. The image is resized to 28 × 28 pixels.
5. Pixel values are normalized between 0 and 1.
6. The trained neural network predicts the digit.
7. The application displays the predicted digit and confidence score.

## Future Improvements

- Improve prediction accuracy.
- Support multiple digit recognition.
- Add a drawing canvas for handwritten digits.
- Improve the user interface.
- Deploy the application online.

## Author

Developed as a Deep Learning project using Python and Flask.

---

**If you find this project useful, consider giving the repository a star!**# digit-prediction
