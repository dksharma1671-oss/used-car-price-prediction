# Used Car Price Prediction using Machine Learning

A final-year B.Tech Machine Learning project that predicts the resale price of a used car.

## Problem Statement
Predict the selling price (in Lakh rupees) of a used car using its age, kilometres driven, fuel type, seller type, transmission and number of owners. This is a **regression** problem.

## Dataset
CarDekho Used Car Dataset (about 15,000 listings from the Indian used car market). The notebook loads it automatically from the internet. If that fails, it asks you to upload a CarDekho CSV file.

## Steps
1. Load and clean the data (price in Lakh, create `car_age`, remove extreme values)
2. Visualise the data (price distribution, age vs price, fuel vs price, heatmap)
3. Encode text columns and split data 80% train / 20% test
4. Train and compare three models
5. Evaluate with R2 score and MAE
6. Predict the price of a new car along with an expected price range

## Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation Metrics
- R2 Score
- Mean Absolute Error (MAE)

## How to Run
1. Open `Used_Car_Price_Prediction.ipynb` in [Google Colab](https://colab.research.google.com).
2. Click **Runtime > Run all**.
3. The dataset loads automatically. No download is needed.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Conclusion
Random Forest generally gives the best results. Car age, kilometres driven and engine power have the biggest effect on resale price. The model also reports a price range using the average error, which is more realistic than a single number.
