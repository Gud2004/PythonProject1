# 🌿 PlantGuardAI - AI-Powered Plant Disease Detection

A deep learning–based web application that predicts plant diseases from leaf images using a Convolutional Neural Network (CNN). Built with TensorFlow, deployed using Streamlit.

[![Live Demo]([https://img.shields.io/badge/Live-Demo-green)](https://plantguardai.streamlit.app](https://pythonproject1-kvabshwgpx859ltgx6socc.streamlit.app/))  


---

## 🌐 Live Demo

🔗 **[Click here to try PlantGuardAI](https://plantguardai.streamlit.app)**  
> Upload a plant leaf image to detect diseases instantly via a responsive and intuitive interface.

---

## 📌 Overview

PlantGuardAI is designed to assist farmers, researchers, and agritech professionals by leveraging AI to accurately detect and classify plant diseases. With over **38 plant classes** and thousands of image samples, the model delivers real-time and highly accurate predictions.

---

## 🎯 Features

- ✅ Built using **Convolutional Neural Networks (CNNs)**
- ✅ Trained on **50,000+ labeled images**
- ✅ Predicts **38 different plant diseases**
- ✅ **~98% validation accuracy**
- ✅ **Live Streamlit demo** for real-time predictions
- ✅ Fully open-source and extensible

---

## 🧠 Model Architecture

- **Framework**: TensorFlow / Keras  
- **Layers**: 3× Conv2D → MaxPooling → Flatten → Dense → Dropout  
- **Activation**: ReLU, Softmax  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy

---

## 📂 Dataset

- **Source**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
- **Contents**: ~54,000 images  
- **Categories**: 38 plant-disease combinations  
- **Preprocessing**: Image resizing, normalization, augmentation

---

## 🧪 Trained Model

🔗 **[Download Trained Model (.h5)](https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/view?usp=drive_link)**

Use this file for local predictions or custom deployment.

---

## 🚀 How to Use

### 📦 Installation

```bash
git clone https://github.com/yourusername/PlantGuardAI.git
cd PlantGuardAI
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
🖼️ Predict a Leaf Image
Upload a plant leaf image

View the predicted disease and class probability

