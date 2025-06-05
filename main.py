import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# Load dataset from train.csv file
df = pd.read_csv("train.csv")

# Features and target
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

# Fill missing values
X = X.fillna(X.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function for Gradio
def predict_price(square_footage, bedrooms, bathrooms):
    try:
        new_data = pd.DataFrame(
            [[square_footage, bedrooms, bathrooms]],
            columns=["GrLivArea", "BedroomAbvGr", "FullBath"]
        )
        prediction = model.predict(new_data)[0]
        return f"Estimated House Price: ₹ {prediction:,.3f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Square Footage (GrLivArea)"),
        gr.Number(label="Bedrooms (BedroomAbvGr)"), 
        gr.Number(label="Bathrooms (FullBath)")
    ],
    outputs=gr.Textbox(label="Predicted House Price"),
    title="🏠 House Price Predictor",
    description="This tool uses a trained Linear Regression model from the Kaggle dataset to predict the price of a house.",
    allow_flagging="never"
)

# Launch the app
interface.launch()
