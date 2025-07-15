# Prodigy_infotech_task3

🐾 Cat and Dog Image Classification using SVM
A computer vision project to classify images of cats and dogs using a Support Vector Machine (SVM) classifier, trained on grayscale image features.

📌 Project Overview
This project demonstrates how to:

  #Load and preprocess image data

  #Extract features by flattening resized grayscale images

  #Train an SVM classifier

  #Evaluate model performance using metrics like accuracy and confusion matrix

  #Predict and visualize test images

Dataset used: Kaggle Dogs vs. Cats

🛠️ Technologies Used
| Tool/Library   | Purpose                                    |
| -------------- | ------------------------------------------ |
| Python         | Core programming language                  |
| OpenCV (`cv2`) | Image preprocessing and resizing           |
| NumPy          | Numerical operations                       |
| scikit-learn   | Machine learning (SVM, evaluation metrics) |
| Matplotlib     | Data visualization and plotting            |


🔍 Model Workflow
   1. Preprocessing:

       #Convert each image to grayscale

       #Resize to 64×64 pixels
  
       #Flatten to a 1D vector

  2. Training:

       #Train a SVC(kernel="linear") model using the image vectors

  3. Evaluation:

       #Classification report

       #Confusion matrix

  4. Visualization:

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


![Capture](https://github.com/user-attachments/assets/2ec9aa20-bec8-4e22-b9bb-4ff491fb7b63)

🚀 How to Run
  1. Download the dataset from Kaggle

  2. Place the train/ folder in the same directory as the notebook

  3. Run the notebook: Cat And Dog Image Classification Using SVM.ipynb


📦 Future Enhancements
   Integrate Histogram of Oriented Gradients (HOG) features

   Use PCA for dimensionality reduction before training

   Replace SVM with Convolutional Neural Networks (CNNs)

   Apply data augmentation for more robust learning


   
