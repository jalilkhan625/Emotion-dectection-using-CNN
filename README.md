# Emotion_Detection_CNN
Data Set Link - https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
# Emotion Detection Using CNN  

## Overview  
This project implements **emotion detection** using a **Convolutional Neural Network (CNN)** with **Keras** and **TensorFlow**. The model classifies facial expressions into different emotion categories such as **happy, sad, angry, surprised, and neutral**.  

## Features  
- Uses **CNN** to extract features from facial expressions.  
- Pretrained **Haar Cascade** classifier for face detection.  
- Trained on labeled datasets for accurate emotion classification.  
- Model saved as `model.h5` for inference.  
- Implemented in **Python** using **Keras, TensorFlow, OpenCV, and NumPy**.  

## Project Structure  
```
📂 Emotion-Detection-Using-CNN
│── emotion-classification-cnn-using-keras.ipynb  # Jupyter Notebook with training & evaluation  
│── haarcascade_frontalface_default.xml           # Pretrained Haar Cascade model for face detection  
│── main.py                                       # Script for real-time emotion detection  
│── model.h5                                      # Trained CNN model  
│── README.md                                     # Project documentation  
```

## Requirements  
Install dependencies using:  
```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

## Usage  

### 1. Train the Model  
Run the Jupyter Notebook:  
```bash
jupyter notebook emotion-classification-cnn-using-keras.ipynb
```

### 2. Real-Time Emotion Detection  
Run the **main.py** script to detect emotions from live webcam feed:  
```bash
python main.py
```

## Model Architecture  
The CNN consists of:  
- **Convolutional layers** for feature extraction  
- **Max-pooling layers** to reduce dimensionality  
- **Fully connected layers** for classification  
- **Softmax activation** for multi-class emotion detection  

## Face Detection  
The **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`) is used to detect faces before passing them to the model.

## Example Output  
When running `main.py`, the webcam detects faces and classifies their emotions in real time.  

## Future Improvements  
- Train on a larger dataset for better accuracy  
- Fine-tune the CNN with **Transfer Learning**  
- Deploy as a web app using Flask or FastAPI  

## Credits  
- OpenCV for face detection  
- TensorFlow/Keras for deep learning  

---
This project provides a foundation for **real-time emotion detection** using deep learning. Feel free to **contribute** or **improve** the model! 🚀

