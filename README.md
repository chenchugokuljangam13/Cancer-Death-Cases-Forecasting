# Cancer Death Case Forecasting in India and Worldwide

## Project Overview
Cancer is a leading cause of death in India and worldwide. This research predicts cancer death cases in India and worldwide using supervised machine learning on data from 1990 to 2017, categorized by age, gender, and region from the Global Burden of Disease Study. Our goal is to provide long-term predictions to help the health department develop strategies to combat cancer.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Results](#results)

## Introduction
This project forecasts cancer death cases in India using three algorithms:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

We found the random forest model performs the best.

## Data Description
We use cancer death case data from the Global Burden of Disease Study, covering the years 1990 to 2017. The data is categorized by:
- Year
- Region
- Type of cancer
- Count of cases

## Project Structure
- **Home**
- **Admin**
  - User details
- **User**
  - Dataset View
  - ML
  - Prediction
- **Register**
  - Registration form

## Installation
To install the required dependencies, run:
'''
pip install -r requirements.txt
'''

## Installation
- Imputing missing values
- Normalizing and scaling features
- Encoding categorical variables

## Model Training
- Splitting the data into training and test sets
- Training each model on the training set
- Saving the trained models
  - XGBoost
 
## Model Evaluation
## Results
