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
<img width="1600" height="738" alt="1000036637" src="https://github.com/user-attachments/assets/38eb895c-ef6d-4e8c-9463-9f270088d0af" />
<img width="1600" height="695" alt="1000036634" src="https://github.com/user-attachments/assets/425414e1-47b4-40bc-9865-5b8b52675d28" />
<img width="1600" height="731" alt="1000036643" src="https://github.com/user-attachments/assets/2da0866b-e072-4cdb-a04a-f37efddf74ac" />
<img width="1600" height="807" alt="1000036640" src="https://github.com/user-attachments/assets/1237a801-02ff-47c9-8559-210949dc0bbe" />
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
