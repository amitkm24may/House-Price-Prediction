# House Price Prediction using Keras Regression

This repository contains the code and data for a house price prediction project using a deep neural network (DNN) built with Keras. The project aims to predict house prices based on various features from the Kaggle dataset: [House Sales Prediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data).

## Project Overview

The project follows a typical machine learning workflow:

1.  **Data Loading and Exploration:**
    * Loading the dataset using Pandas.
    * Performing exploratory data analysis (EDA) using Pandas, NumPy, Seaborn, and Matplotlib.
    * Visualizing data distributions and relationships between features.
    * Analyzing geographical data (latitude and longitude).
2.  **Feature Engineering:**
    * Dropping irrelevant columns (e.g., 'id').
    * Extracting month and year from the 'date' column.
    * Removing less useful columns such as zipcode.
    * Handling potential outliers.
3.  **Data Preprocessing:**
    * Splitting the data into training and testing sets.
    * Scaling features using MinMaxScaler.
4.  **Model Building:**
    * Building a DNN model using Keras with multiple dense layers and ReLU activation.
5.  **Model Training:**
    * Compiling the model with the Adam optimizer and mean squared error (MSE) loss.
    * Training the model and visualizing the loss curves.
6.  **Model Evaluation:**
    * Generating predictions on the test set.
    * Calculating evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and explained variance score.
    * Visualizing predicted vs. actual prices.

## Dataset Source

The dataset used in this project is from Kaggle:

* [House Sales Prediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data)

## Repository Contents

* `kc_house_data.csv`: The dataset used for training and testing.
* `house_price_prediction.ipynb`: Jupyter Notebook containing the code for data analysis, model building, and evaluation.
* `README.md`: This file.

## Dependencies

* Python 3.x
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* TensorFlow/Keras

To install the required libraries, you can use pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
