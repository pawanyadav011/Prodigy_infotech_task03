🐾 Cat and Dog Image Classification using SVM A computer vision project to classify images of cats and dogs using a Support Vector Machine (SVM) classifier, trained on grayscale image features.

📌 Project Overview This project demonstrates how to:

#Load and preprocess image data

#Extract features by flattening resized grayscale images

#Train an SVM classifier

#Evaluate model performance using metrics like accuracy and confusion matrix

#Predict and visualize test images

Dataset used: Kaggle Dogs vs. Cats

🛠️ Technologies Used

Tool/Library	Purpose
Python	Core programming language
OpenCV (cv2)	Image preprocessing and resizing
NumPy	Numerical operations
scikit-learn	Machine learning (SVM, evaluation metrics)
Matplotlib	Data visualization and plotting
🔍 Model Workflow

Preprocessing:

#Convert each image to grayscale

#Resize to 64×64 pixels

#Flatten to a 1D vector

Training:

#Train a SVC(kernel="linear") model using the image vectors

Evaluation:

#Classification report

#Confusion matrix

Visualization:

#Display predictions for random test images

🧠 SVM Model

#Type: Binary classification

#Kernel: Linear

#C parameter: 1.0

#Features: Raw pixel intensities (flattened grayscale images)

Note: SVMs are sensitive to high dimensionality. Therefore, we use a reduced image size and subset of the dataset.

📈 Results

#Accuracy: ~85% (on a small subset of training data)

#Performance may vary based on image size, feature representation, and number of samples

🖼️ Sample Output

Capture

🚀 How to Run

Download the dataset from Kaggle

Place the train/ folder in the same directory as the notebook

Run the notebook: Cat And Dog Image Classification Using SVM.ipynb

📦 Future Enhancements Integrate Histogram of Oriented Gradients (HOG) features

Use PCA for dimensionality reduction before training

Replace SVM with Convolutional Neural Networks (CNNs)

Apply data augmentation for more robust learning