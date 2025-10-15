# Skin Lesion Detection with Deep Learning
A deep learning project that classifies skin lesions as **benign** or **malignant** using a RestNet50 CNN and a custom focal loss function to handle class imbalance. Includes Grad-CAM visualizations for model explainability and a Streamlit web app for real-time predictions.  

### View the app here: -> [https://skin-lesion-class.streamlit.app/](https://skin-lesion-class.streamlit.app/)


## Project Overview

This project aims to support the detection of skin cancer by leveraging deep learning to analyze dermoscopic images.  
A ResNet50 backbone was fine-tuned on the 
[HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), and the model was trained with focal loss to improve performance on underrepresented malignant cases.  

To make the model interpretable, I implemented Grad-CAM (Gradient-weighted Class Activation Mapping), which highlights the regions of each image that are most influential to the model's decision.  

A Streamlit web app was developed to enable users to upload lesion images, view classification results, and interactively visualize Grad-CAM heatmaps.  

## Tech Stack
- Deep Learning: TensorFlow, Keras (ResNet50, custom focal loss)
- Data Processing: NumPy, Pandas
- Visualization: Matplotlib, OpenCV
- App Deployment: Streamlit
- Environment: Google Colab + Google Drive (mount)

## How to Run
Upload a skin lesion image (`.jpg`, `.jpeg`, or `.png`) and view:
- Model prediction (Benign or Malignant)
- Prediction accuracy
- Grad-CAM heatmap
- Overlay visualization

> ⚠️ MEDICAL DISCLAIMER: This is a research prototype only. NOT for medical diagnosis. Always consult healthcare professionals for medical advice.

### Author: Rayyan Huda

[rayyanhuda.com](https://rayyanhuda.com/)  

[LinkedIn](https://www.linkedin.com/in/rayyanhuda/)
