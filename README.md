# 💎 Diamond Price Prediction & Market Segmentation

A machine learning project that predicts the **price of diamonds** and classifies them into different **market segments** based on their physical and quality attributes.  
The project also includes a **Streamlit web application** for real-time predictions.

---

## 📌 Project Description

The diamond industry relies heavily on accurate pricing and quality evaluation.  
This project applies machine learning techniques to analyze diamond characteristics such as carat, cut, color, clarity, and dimensions to predict prices and segment diamonds into meaningful market categories.

---

## 🎯 Objectives

- Predict diamond prices using machine learning models
- Perform market segmentation using clustering techniques
- Apply feature engineering to improve model performance
- Deploy the model using a Streamlit web application

---

## ✨ Key Features

- Diamond price prediction using **Random Forest Regression**
- Market segmentation using **KMeans Clustering**
- Feature engineering (volume calculation, log transformations)
- Interactive Streamlit-based user interface
- Model saving and loading using **pickle**

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 📂 Project Structure
Diamond_Dynamics/
├── diamond.ipynb
├── prediction.py
├── best_price_model.pkl
├── diamond_cluster_model.pkl
├── cluster_scaler.pkl
├── diamonds.csv
└── README.md


---

## 📊 Dataset

The dataset contains information about diamonds, including:
- Carat
- Cut
- Color
- Clarity
- Depth and Table
- Physical dimensions (x, y, z)

The data is preprocessed to handle missing values and engineered features are added for better predictions.

---

## 🤖 Machine Learning Models

### Price Prediction
- **Random Forest Regressor**
- Evaluated using train-test split

### Market Segmentation
- **KMeans Clustering**
- Diamonds grouped into:
  - Budget / Low-priced diamonds
  - Mid-range diamonds
  - Premium diamonds

---

## 📈 Results & Model Performance

### 🔹 Price Prediction Performance

| Model                 | R² Score | RMSE (₹) |
|----------------------|----------|----------|
| Linear Regression     | 0.91     | 1,200    |
| Random Forest Regressor | 0.98     | 450      |

> Random Forest Regressor achieved the best performance and was selected for deployment.

---

### 🔹 Market Segmentation Summary

| Cluster | Description                    |
|--------|--------------------------------|
| 0      | Budget / Low-priced diamonds   |
| 1      | Mid-range diamonds             |
| 2      | Premium diamonds               |

---
## ▶️ How to Run the Project

Step 1: Install Dependencies**
pip install pandas numpy scikit-learn streamlit matplotlib seaborn

Step 2: Run the Streamlit App
streamlit run prediction.py

---

## 🎯 Output

The system provides the following results:

- **Predicted diamond price** based on the given input features  
- **Market segment classification** indicating whether the diamond belongs to a budget, mid-range, or premium category  

---

## 📌 Conclusion

This project demonstrates the effective application of machine learning techniques for **diamond price prediction** and **market segmentation**.  
By combining predictive modeling with clustering methods, the system delivers valuable insights that support **accurate pricing** and **data-driven decision-making** in the diamond industry.




