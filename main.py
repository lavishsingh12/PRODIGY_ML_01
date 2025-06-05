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
        # Check if any field is empty
        if not square_footage or not bedrooms or not bathrooms:
            return "<span style='color:orange;'>⚠️ Please fill in all fields before submitting.</span>"

        # Try converting inputs to proper types
        square_footage = float(square_footage)
        bedrooms = int(bedrooms)
        bathrooms = int(bathrooms)

        new_data = pd.DataFrame(
            [[square_footage, bedrooms, bathrooms]],
            columns=["GrLivArea", "BedroomAbvGr", "FullBath"]
        )
        prediction = model.predict(new_data)[0]
        return f"""
        <div style='text-align:center; font-size: 28px; color: green; font-weight: bold;'>
            Estimated House Price: ₹ {prediction:,.3f}
        </div>
        """
    except ValueError:
        return "<span style='color:red;'>❌ Please enter valid numeric values only.</span>"
    except Exception as e:
        return f"<span style='color:red;'>Unexpected error: {str(e)}</span>"

# Gradio UI
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Textbox(label="Square Footage (GrLivArea)", placeholder="0"),
        gr.Textbox(label="Bedrooms (BedroomAbvGr)", placeholder="0"),
        gr.Textbox(label="Bathrooms (FullBath)", placeholder="0")
    ],
    outputs=gr.HTML(label="Predicted House Price"),
    title="🏠 House Price Predictor",
    description="This tool uses a trained Linear Regression model on the Ames Housing dataset to predict the price of a house.",
    allow_flagging="never"
)

interface.launch()
