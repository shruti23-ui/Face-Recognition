# Face Recognition using K-Nearest Neighbors (KNN)

## Description
This project implements a real-time face recognition system using Python, OpenCV, and a custom K-Nearest Neighbors (KNN) algorithm. The system captures face data, preprocesses it, and recognizes faces in real-time using a pre-trained dataset.

## Features
- **Real-time Face Detection:** Utilizes OpenCV's Haarcascade classifier for detecting faces in real-time.
- **Custom KNN Algorithm:** Implements a simple yet effective KNN algorithm for face classification.
- **Dynamic Dataset Loading:** Supports loading and processing of face datasets stored in `.npy` files.
- **Interactive Output:** Displays detected faces and their labels on the video feed.

## How It Works
1. **Dataset Preparation:**
   - Loads face data (`.npy` files) from a specified folder.
   - Assigns labels to faces dynamically based on the dataset.
2. **Training:**
   - Combines face data and labels into a training set.
3. **Real-Time Recognition:**
   - Captures frames from the webcam.
   - Detects faces in each frame.
   - Preprocesses face regions and uses the KNN algorithm to classify them.
   - Displays the detected faces with their corresponding labels.

## Dependencies
- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-knn.git
   cd face-recognition-knn
2. Install the required dependencies:
    ```bash
    pip install numpy opencv-python
3. Ensure the haarcascade_frontalface_alt.xml file is in the project directory.
4. Prepare your dataset:
  - Place `.npy` files in the face_dataset/ directory.
5. Run the face recognition script:
    ```bash
    python face_recognition.py
6. Press q to quit the application.

# File Structure
- `face_recognition.py`: Main script for face recognition.
- `face_dataset/`: Folder to store the face dataset (.npy files).
- `haarcascade_frontalface_alt.xml`: Pre-trained face detection model.

# Future Enhancements
- Integrate a more advanced classification model like SVM or a neural network.
- Add functionality for real-time dataset creation and updates.
- Improve face detection by utilizing deep learning models like SSD or YOLO.
- Enhance performance with GPU acceleration.

# Acknowledgments
This project was built as a simple demonstration of KNN-based face recognition. Thanks to OpenCV for providing robust tools for image processing and NumPy for efficient numerical operations.
Feel free to contribute or raise issues!

    
