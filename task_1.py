import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset from train.csv file
df = pd.read_csv("train.csv")

# Features and target
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

# Fill missing values if any
X = X.fillna(X.mean())

# Split dataset into Traning and testing parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Take user input
square_footage = float(input("Enter square feet of the house (sq.ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Prepare input for prediction.
import pandas as pd

new_house = pd.DataFrame(
    [[square_footage, bedrooms, bathrooms]],
    columns=["GrLivArea", "BedroomAbvGr", "FullBath"],
)
predicted_price = model.predict(new_house)[0]

# Print result with 3 decimal places.
print(f"Predicted Price for the house: {predicted_price:.3f}")
