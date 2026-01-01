Bone Fracture Detection with TensorFlow & FastAPI

Detect bone fractures in X-ray images using a custom-built deep learning model with a simple web interface.




📝 Project Overview

This project demonstrates how to build an object detection model from scratch to localize bone fractures in X-ray images. The model predicts bounding boxes around fractures, enabling doctors or students to visualize fracture locations.

It includes:

A TensorFlow/Keras model trained from scratch on a fracture dataset

A visualization module to inspect predictions

A FastAPI web app to upload X-ray images and get predictions in real time

Optional Live deployment via ngrok for sharing or demoing

This project is meant for educational purposes, not for clinical use.

🔬 Dataset

The model was trained on a publicly available dataset (from Kaggle) with X-ray images of bones.

Training images: X-ray images of bones with fractures

Labels: bounding boxes normalized between 0 and 1, representing fracture locations

Note: The labels were originally in YOLO notation, so a small preprocessing step was applied to convert them into [xmin, ymin, xmax, ymax] format suitable for training the model.

Note: The dataset may include augmented images to improve model generalization.

🏗 Model Architecture

The custom model is a convolutional neural network (CNN) designed to predict bounding boxes [xmin, ymin, xmax, ymax]:

Conv2D + MaxPooling layers to extract features

GlobalAveragePooling2D to reduce feature maps

Dense layers to predict 4 bounding box coordinates

Activation: Sigmoid on the output for normalized box coordinates

Loss function: Huber loss, robust to outliers in bounding box coordinates.

Optimizer: Adam with learning rate scheduling and optional early stopping.

🖼 Visualization

Two utility functions help visualize predictions:

visualize_prediction(model, img, true_box=None) → Single image prediction

visualize_many_grid(model, X_val, y_val, n=20, cols=5) → Grid of multiple images

Green/red/lime rectangles indicate ground truth vs predicted boxes.

⚡ FastAPI Web App

The project includes a simple GUI where you can:

Upload an X-ray image

Get the predicted fracture bounding box overlaid on the image

Download or view the results directly in the browser

Installation & running locally:

# 1. Clone the repo
git clone https: git clone https://github.com/KareemSoltan/Bone-fracture-detection-.git

cd bone-fracture-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the FastAPI app
uvicorn app:app --reload


Access the GUI:
Navigate to http://127.0.0.1:8000
 in your browser.

Optional: use ngrok to share your app publicly.







Use early stopping and learning rate scheduler for better convergence

Loss is Huber loss (robust to outliers in bounding boxes)

💡 Key Learnings

How to build a from-scratch CNN for bounding box regression

How to use visualizations to debug object detection models

How to deploy a ML model as a simple web app using FastAPI

How to handle real-world medical data for educational purposes

⚠️ Disclaimer

This project is for educational purposes only.
It is not a substitute for professional medical diagnosis.
