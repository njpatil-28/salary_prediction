# 💼 AI Salary Class Predictor

A sleek and intelligent web application built with **Streamlit** that predicts an employee's **salary class** using machine learning. Users can enter personal and professional information like age, gender, education level, job title, and years of experience to get an instant prediction about their salary category.

---

## 🚀 Features

- 🔍 **Predicts salary class** based on trained ML model.
- 🌗 **Toggle between Light and Dark themes** with elegant styling.
- 📊 Displays model accuracy, prediction count, and inference speed.
- 💬 Real-time **emoji feedback** based on predicted salary class.
- ✅ Confidence percentage (simulated) to indicate model assurance.
- 📦 Encoders and model loaded using `pickle` and cached for performance.

---

## 📂 File Structure
  
  salary_prediction_app/
  │
  ├── app.py # Main Streamlit application
  ├── salary_bagging_model.pkl # Trained ML model
  ├── scaler.pkl # Standard scaler for features
  ├── gender1_encoder.pkl # Label encoder for gender
  ├── education_level_encoder.pkl # Label encoder for education
  ├── job_title_encoder.pkl # Label encoder for job titles
  ├── README.md # You're here!

  

## 🧠 Model Info

- **Algorithm**: Bagging Regressor
- **Accuracy**: ~90.2%
- **Target**: Predict the numeric salary class (actual salary value).
- **Input Features**:
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience
