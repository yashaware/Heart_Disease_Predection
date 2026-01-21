# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview
Heart disease is one of the leading causes of mortality worldwide. Early prediction can help in taking preventive measures and improving patient outcomes.

This project focuses on predicting whether a patient will develop **heart disease within 10 years** using **machine learning classification models**. The complete end-to-end pipeline includes data preprocessing, exploratory data analysis, feature selection, handling class imbalance, model training, evaluation, and prediction.

The dataset used in this project is the **Framingham Heart Study dataset**.

---

## 🎯 Project Objectives
- Perform Exploratory Data Analysis (EDA) on health-related data
- Handle missing values and outliers effectively
- Identify the most important features contributing to heart disease
- Address class imbalance in the dataset
- Train and evaluate multiple machine learning models
- Compare model performance using appropriate evaluation metrics
- Predict heart disease risk for new patient data

---

## 📂 Dataset Information
- **Dataset**: Framingham Heart Study
- **Records**: ~4,240 (after cleaning ~3,749)
- **Target Variable**: `TenYearCHD`  
  - `0` → No heart disease  
  - `1` → Heart disease within 10 years

### 🔢 Features Description
| Feature | Description |
|------|------------|
| age | Age of the patient |
| male | Gender (1 = Male, 0 = Female) |
| currentSmoker | Smoking status |
| cigsPerDay | Cigarettes smoked per day |
| BPMeds | Blood pressure medication |
| prevalentStroke | History of stroke |
| prevalentHyp | Hypertension |
| diabetes | Diabetes status |
| totChol | Total cholesterol |
| sysBP | Systolic blood pressure |
| diaBP | Diastolic blood pressure |
| BMI | Body Mass Index |
| heartRate | Heart rate |
| glucose | Glucose level |
| TenYearCHD | Target variable |

---

## 🔧 Data Preprocessing
- Checked and removed **duplicate records**
- Handled **missing values** by:
  - Dropping highly missing features
  - Removing rows with remaining null values
- Detected and removed **outliers** (notably in cholesterol levels)
- Applied **MinMaxScaler** for feature normalization
- Performed feature correlation analysis using heatmaps

---

## 📊 Exploratory Data Analysis (EDA)
- Distribution plots for all numerical features
- Correlation heatmaps to identify feature relationships
- Pair plots to visualize feature interactions
- Boxplots for detecting outliers
- Class distribution analysis showed a **highly imbalanced dataset**

---

## ⭐ Feature Selection
- Used **SelectKBest with Chi-Square test**
- Identified top contributing features such as:
  - Systolic Blood Pressure (sysBP)
  - Glucose
  - Age
  - Total Cholesterol
  - Cigarettes per day
  - Diastolic BP
  - Hypertension
  - Diabetes

Only the **top 10 most important features** were used for final modeling.

---

## ⚖️ Handling Class Imbalance
- Original class ratio: **~5.5 : 1**
- Applied **random under-sampling**
- Created a balanced dataset for fair model training

---

## 🧠 Machine Learning Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Hyperparameter tuning using **GridSearchCV**
- Pipeline-based model comparison

---

## 📈 Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

### 🏆 Best Model
- **K-Nearest Neighbors (KNN)**
- Accuracy ≈ **80%**
- ROC-AUC ≈ **63.5%**
- Demonstrated strong balance between precision and recall

---

## 🧪 Results Summary
- Logistic Regression provided baseline performance
- Decision Tree achieved high recall
- KNN performed best overall after tuning
- Visualization of confusion matrices helped interpret misclassifications

---

## 🔮 Prediction System
- Implemented a **command-line questionnaire**
- Accepts patient health parameters
- Predicts whether the patient is likely to develop heart disease within 10 years

---

## 🛠️ Technologies & Tools Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶️ How to Run the Project
1. Clone the repository
2. Place `framingham.csv` in the project directory
3. Open the Jupyter Notebook
4. Run all cells sequentially
5. Use the prediction questionnaire for new inputs

---

## 📈 Future Enhancements
- Apply SMOTE instead of undersampling
- Use ensemble models (Random Forest, XGBoost)
- Deploy the model using Flask or FastAPI
- Build a web-based prediction interface
- Improve ROC-AUC through feature engineering

---

## 👤 Author
**Yash Bhupesh Aware**  
Engineering Student | Machine Learning & Data Science Enthusiast

---

## 📜 License
This project is intended for educational and academic purposes only.
