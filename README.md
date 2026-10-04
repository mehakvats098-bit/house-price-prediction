# 🏠 House Price Prediction

A machine learning regression project that predicts house prices based on property characteristics such as area, bedrooms, bathrooms, age, location, distance from the city, and parking availability.

The project includes exploratory data analysis, data preprocessing, Linear Regression model training, model evaluation, visualization, and an interactive Streamlit dashboard.

## 🎯 Project Objective

The goal of this project is to build a machine learning model that can estimate the price of a house based on its features.

### Key Questions

* How does house area affect price?
* How does location influence house prices?
* Does the number of bedrooms and bathrooms affect price?
* Does distance from the city affect property value?
* How accurately can a machine learning model predict house prices?

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and analysis
* **Matplotlib** — Data visualization
* **Scikit-learn** — Machine learning
* **Joblib** — Model saving and loading
* **Streamlit** — Interactive dashboard
* **Git & GitHub** — Version control

## 🤖 Machine Learning

### Algorithm

**Linear Regression**

### Problem Type

**Supervised Learning — Regression**

### Target Variable

`Price_Lakh`

The model uses the following features:

* `Area_sqft`
* `Bedrooms`
* `Bathrooms`
* `Age_Years`
* `Distance_to_City_km`
* `Location`
* `Parking`

Categorical features such as location and parking were converted into numerical features using one-hot encoding.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning & Inspection
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Price Prediction
   ↓
Model Evaluation
   ↓
Streamlit Dashboard
```

## 📊 Exploratory Data Analysis

The project analyzes:

* Overall house price distribution
* Relationship between area and price
* Average price by location
* Average price by number of bedrooms
* Actual vs predicted prices

### Charts

The project generates the following visualizations:

* `price_distribution.png`
* `area_vs_price.png`
* `price_by_location.png`
* `price_by_bedrooms.png`
* `actual_vs_predicted.png`

## 📈 Model Evaluation

The model is evaluated using:

* **MAE — Mean Absolute Error**
* **MSE — Mean Squared Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**

The evaluation results are stored in:

```text
model/evaluation_report.txt
```

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard with three main sections:

### 🏠 Price Prediction

Users can enter property details such as:

* Area
* Bedrooms
* Bathrooms
* Age
* Distance from city
* Location
* Parking availability

The dashboard then provides an estimated house price.

### 📊 Market Analysis

Users can select a location and explore:

* Average property price
* Price difference from the overall market
* Minimum and maximum prices
* Location-based visualizations

### 🤖 Model Performance

Displays:

* MAE
* MSE
* RMSE
* R² Score
* Actual vs predicted price visualization
* Model information

## 📁 Project Structure

```text
house-price-prediction/
│
├── data/
│   └── house_prices.csv
│
├── charts/
│   ├── price_distribution.png
│   ├── area_vs_price.png
│   ├── price_by_location.png
│   ├── price_by_bedrooms.png
│   └── actual_vs_predicted.png
│
├── model/
│   ├── house_price_model.pkl
│   ├── feature_columns.pkl
│   └── evaluation_report.txt
│
├── analysis.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/mehakvats098-bit/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

```bash
python analysis.py
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

## 📌 Dataset

The dataset contains **40 house records** with property, location, and pricing information.

The dataset is a **synthetic dataset created for educational and portfolio purposes** and should not be treated as real-world property pricing data.

## 💡 Key Learning Outcomes

Through this project, I practiced:

* Regression-based machine learning
* Feature selection
* Categorical data encoding
* Train-test splitting
* Model evaluation
* Data visualization
* Saving and loading ML models
* Building interactive ML dashboards with Streamlit
* Organizing an end-to-end machine learning project

## 👩‍💻 Author

**Mehak**
AIML Student

GitHub: `mehakvats098-bit`
