# Cancer death Case Forecasting in India and worldwide.

## Project Overview
Cancer is a leading cause of death in India and worldwide. This research predicts cancer death Case in India and worldwide using supervised machine learning on data from 1990 to 2017, categorized by age, gender, and region from the Global Burden of Disease Study. Our goal is to provide long-term predictions to help the health department develop strategies to combat cancer.

## Table of Content
1. [Introduction](#Introduction)
2. [Data Description](#Data-description)
3. [Project Structure](#Project-structure)
4. [Installation](#Installation)
5. [Data Preprocessing](#Data-preprocessing)
6. [Model Training](#Model-training)
7. [Model Evaluation](#Model-evaluation)
8. [Results](#Results)



## Introduction
This project Forecasts cancer death cases in India using three algorithms: 
linear regression
decision tree regression
random forest regression
We found the random forest model performs the best.

## Data Description
We use cancer death case data from the Global Burden of Disease Study, covering the years 1990 to 2017. The data is categorized by:
- Year
- Region
- Type of cancer
- Count of cases

## Project structure
- Home
- Admin
  - user details
- User
  - DatasetView
  - Ml
  - Prediction
- register
  - registeration form

## Installation
- installing requirements.txt

## Data Preprocessing
- Imputing missing values
- Normalizing and scaling features
- Encoding categorical variables

## Model Training
- Splitting the data into training and test sets
- Training each model on the training set
- Saving the trained models
  - XGBoost
