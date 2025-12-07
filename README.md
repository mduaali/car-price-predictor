# Car Price Predictor

**Car Price Predictor** is a personal project built with Django and Python to practice AI and machine learning skills. It uses a trained RandomForest model to predict used car prices in **Indian Rupees (INR)** based on car details like year, kilometers driven, fuel type, seller type, transmission, and ownership history.

## Features

- Built with **Django** for a simple web interface
- Predicts car prices in INR
- Trained with **RandomForestRegressor** from scikit-learn
- Handles categorical and numerical features with preprocessing pipelines
- Includes interactive form to input car details

## Dataset

The dataset used for training the model was sourced from [Kaggle](https://www.kaggle.com/datasets) (Car details and selling prices dataset). It contains information like:

- Car name, year, and kilometers driven
- Fuel type, seller type, transmission
- Ownership history
- Selling price in INR

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/car-price-predictor.git
cd car-price-predictor
