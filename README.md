# NPDR-Detection-System-Project-
# 🩺 NPDR Detection System using Deep Learning

## 📌 Project Overview
The NPDR (Non-Proliferative Diabetic Retinopathy) Detection System is a deep learning-based web application that detects the early stages of diabetic retinopathy from retinal fundus images. The project helps in early diagnosis, allowing timely treatment and reducing the risk of vision loss.

## 🚀 Features
- Upload retinal fundus images
- Image preprocessing using OpenCV
- Deep Learning-based prediction
- User-friendly Streamlit interface
- Fast and accurate disease detection

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- PIL (Pillow)

## 📂 Project Structure

NPDR-Detection-System-Project/
│── app5.py                 # Streamlit Application
│── Preprocessing.py        # Image preprocessing
│── Model building          # Deep Learning model
│── README.md
``
## ▶️ How to Run

```bash
git clone https://github.com/YourUsername/NPDR-Detection-System-Project.git

cd NPDR-Detection-System-Project

pip install -r requirements.txt

streamlit run app5.py

## 📊 Model Output

The model classifies retinal fundus images into:

- ✅ No NPDR
- ⚠️ Mild NPDR
- ⚠️ Moderate NPDR
- ⚠️ Severe NPDR
- 🔴 Proliferative Diabetic Retinopathy

with proper reason and its give report 

## 📸 Output

### Home Page
![Home Page](home.png)

### Prediction Result
![Prediction](prediction.png)
![Prediction](reason.png)
![Prediction](score.png)

### Example Prediction

| Input Image | Prediction |
|-------------|------------|
| Retinal Fundus Image | Moderate NPDR |

## 🎯 Future Improvements
- Improve model accuracy
- Deploy on cloud
- Mobile application support
- Multi-class classification

## 👩‍💻 Author
Vaishnavi Pawar
Artificial Intelligence & Machine Learning Engineer
