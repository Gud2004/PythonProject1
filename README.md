🌿 PlantGuardAI - Deep Learning Powered Plant Disease Detection
An end-to-end image classification system for real-time plant disease detection using Convolutional Neural Networks (CNN) and Streamlit.

🚀 Live Demo
🔗 Try the App on Streamlit

Upload a plant leaf image to detect its disease instantly.

📌 About the Project
PlantGuardAI is an AI-powered web application that identifies plant diseases from leaf images using a custom-built deep learning model. This project aims to empower farmers, researchers, and agriculturists with early disease detection to improve crop health and yield.

🎯 Key Features
✅ Trained on PlantVillage dataset with 38 classes
✅ Achieved ~98% accuracy on validation data
✅ Real-time prediction via Streamlit Web App
✅ Supports custom image uploads
✅ Deployed and production-ready
✅ Clean, intuitive UI for ease of use

🧠 Model Details
Framework: TensorFlow / Keras

Model: Convolutional Neural Network (CNN)

Architecture: 3 Conv2D + MaxPooling2D layers, followed by Dense layers

Loss Function: Categorical Crossentropy

Optimizer: Adam

Training Accuracy: 99%

Validation Accuracy: ~98%

📊 Dataset
Source: Kaggle - PlantVillage Dataset

Size: 50,000+ labeled images

Classes: 38 plant species and disease types

Format: RGB images categorized by plant type and disease

🧪 Trained Model
📥 Download Trained Model (.h5)

🛠️ Installation & Usage
⚙️ Requirements
bash
Copy
Edit
pip install -r requirements.txt
▶️ Run Locally
bash
Copy
Edit
streamlit run app.py
