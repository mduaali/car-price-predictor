import pandas as pd
from pathlib import Path

# path to the csv
data_path = Path("data/car_details.csv")

# load dataset
df = pd.read_csv(data_path)

# peek head and get how many rows and columns
#print("Shape:", df.shape)
#print(df.head())

# basic cleaning

# 1) drop duplicate rows
df = df.drop_duplicates()

# 2) drop rows where the target is missing or zero
df = df[df["selling_price"].notna()]
df = df[df["selling_price"] > 0]

# 3) drop columns we do NOT want as features
#    name of the car doesn't help the model much and causes too many categories
df = df.drop(columns=["name"], errors="ignore")

# split features into numerical vs. categorial for preprocessing

target = "selling_price"
X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include= ["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

# building pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# numeric pipeline, fill missing val with median + scale big nums
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# categorical pipeline, fill missing w most frequent + encode into nums so model understands words
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# combining both pipelines into a preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# importing model and doing train / test split to learn patters and check if it generalizes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(x_train.shape[0], "train samples")
#print(x_test.shape[0], "test samples")

# create full pipeline: preprocessing + model
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200,random_state=42))
])

# train model
model_pipeline.fit(x_train, y_train)

# importing metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# predictions on test set
y_pred = model_pipeline.predict(x_test)

# calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# save entire job pipeline
import joblib
from pathlib import Path

#to check if models folder exists to save trained pipeline
Path("models/").mkdir(parents=True, exist_ok=True)

#save trained pipeline
joblib.dump(model_pipeline, "models/car_price_model.joblib")
