import streamlit as st
import pandas as pd
import joblib  # To load the model

# Load your trained model (make sure to save it as a .pkl file after training)
model = joblib.load('best_model.pkl')  # Replace with your model filename

# Title of the app
st.title("Churn Prediction App")

# Input fields for the features
st.header("Input Features")

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, step=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Credit card (automatic)", "Mailed check", "Bank transfer (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.01)

# Convert input data into a DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [1 if partner == "Yes" else 0],
    'Dependents': [1 if dependents == "Yes" else 0],
    'tenure': [tenure],
    'PhoneService': [1 if phone_service == "Yes" else 0],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
})

# Encode categorical variables (one-hot encoding)
input_data_encoded = pd.get_dummies(input_data, drop_first=True)


# Ensure all columns required by the model are present (handle missing columns)
for column in model.feature_names_in_:  # Use feature_names_in_ from your model to get the required features
    if column not in input_data_encoded.columns:
        input_data_encoded[column] = 0

# Reorder the columns to match the training set
input_data_encoded = input_data_encoded[model.feature_names_in_]

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data_encoded)
    prediction_proba = model.predict_proba(input_data_encoded)[:, 1]
    churn_status = "Churn" if prediction[0] == 1 else "No Churn"
    st.success(f"Prediction: {churn_status}")
    st.write(f"Probability of Churn: {prediction_proba[0]:.2f}")

