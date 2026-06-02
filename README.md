# AI Sign Language Detection

## Overview

AI Sign Language Detection is a machine learning project that recognizes hand signs and converts them into text in real time. The project uses MediaPipe for hand landmark detection and a trained machine learning model for sign classification. A Streamlit web application provides an interactive interface for users to perform sign language recognition through a webcam.

## Features

* Real-time hand detection using MediaPipe
* Sign language alphabet recognition
* Machine learning-based prediction
* User-friendly Streamlit interface
* Live webcam support
* Fast and accurate gesture classification

## Technologies Used

* Python
* Streamlit
* OpenCV
* MediaPipe
* Scikit-learn
* NumPy
* Joblib

## Project Structure

AI-Sign-Language-Detection/

├── app.py

├── train_model.py

├── sign_language_model.p

├── requirements.txt

├── README.md

└── dataset/

## Dataset

The model was trained using an American Sign Language (ASL) alphabet dataset containing hand gesture images representing different alphabet classes.

Dataset source can be added here if publicly available.

## Installation

1. Clone the repository

```bash
git clone (https://github.com/SonalJC/AI-Sign-Language-Detection.git)
```

2. Navigate to the project folder

```bash
cd AI-Sign-Language-Detection
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

## Model Training

To train the model again:

```bash
python train_model.py
```

This will generate a trained model file used for sign language prediction.

## Future Improvements

* Support for words and sentences
* Text-to-speech conversion
* Multiple hand detection
* Improved model accuracy
* Support for additional sign languages

## Author

Sonal Chauhan

B.Tech Student | Machine Learning & AI Enthusiast
