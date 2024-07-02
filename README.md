# Cancer Death Case Forecasting in India and Worldwide

## Project Overview
Cancer is a leading cause of death in India and worldwide. This research predicts cancer death cases in India and worldwide using supervised machine learning on data from 1990 to 2017, categorized by age, gender, and region from the Global Burden of Disease Study. Our goal is to provide long-term predictions to help the health department develop strategies to combat cancer.

## Table of Contents
1. [Introduction](#Introduction)
2. [Data Description](#Data-description)
3. [Project Structure](#Project-structure)
4. [Installation](#Installation)
5. [Dataset view](#Dataset-view)
6. [Regression ML](#Regression-ML)
7. [Forecasting](#Forecasting)
8. [Results](#Results)

## Introduction
This project forecasts cancer death cases in India using three algorithms:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

We found the random forest model performs the best.

## Data Description
We use cancer death case data from the Global Burden of Disease Study, covering the years 1990 to 2017. The data is categorized by:
- Region
- Year
- Type of cancer
- Count of deathcases 

## Project Structure
- Home
- User
  - Dataset
  - [Regression](#Regression)
    - read data
    - tarin-test-split data
    - run models(regression algorithms)
  - Forecast
    - read data
    - forecasting
- Admin Login
  - User details
- Register
  - Registration form

## Installation
- **Install Django**
  - `pip install django`
- **Installing requirements.txt**
  - `pip install -r requirements.txt`
    - Use the above command to install the packages listed in the requirements file.
- **Runserver**
  - `py manage.py runserver`
    - Use the above command to start the server.

## Dataset veiw
- Data Collection
- filter data
  - filter data based on the country code, rename columns, drop unnecessary columns, and convert the DataFrame to HTML.
- Render Template

## Regression ML
- Read Data:
  - Read the data file, filter data based on the country code, and prepare the data for regression.
- Train-Test Split:
  - Split the data into training and testing sets.
- Run Models:
  - Run multiple regression models i.e.,(Linear Regression, Decision Tree, Random Forest, Polynomial Regressor).
  - Evaluate the each model by using R² score, mean squared error, Mean Absolute Error, Explained Variance Score and Root Mean Squared Error
- Render Template

## regression Models
By using the machine learning and regression models we can forecast the future predictions
- Linear Regression
  - Linear regression fits a linear relationship between the input features (X) and the target variable (y).
- Decision Tree
  - Decision trees split the data into subsets based on feature values, creating a tree-like model of decisions.
- Random Forest
  - Random forest is an ensemble method that fits multiple decision trees on various sub-samples of the dataset and averages the predictions to improve accuracy and control over-fitting.
- Polynomial Regression
  - Polynomial regression fits a polynomial equation to the data, capturing non-linear relationships.

## Forecasting
- Read the CSV file, filter data based on the country code, and prepare the data for forecasting.
- Perform future predictions on the data, format the results, and convert the DataFrame to HTML.

## results
