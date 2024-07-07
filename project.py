import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
insurance = pd.read_csv("/home/vs/Downloads/insurance (1).csv")

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
insurance['sex'] = le.fit_transform(insurance['sex'])
insurance['smoker'] = le.fit_transform(insurance['smoker'])

# Define features and target variable
X = insurance[['age', 'bmi', 'children', 'sex', 'smoker']]
y = insurance['charges']

# Initialize the Gradient Boosting model
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbm_model.fit(X, y)

# Streamlit app
st.title("Insurance Charge Prediction")

# Input features
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("southeast", "southwest", "northeast", "northwest"))

# Encode categorical variables
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

# Make prediction
input_data = {'age': age, 'bmi': bmi, 'children': children, 'sex': sex_encoded, 'smoker': smoker_encoded}
input_df = pd.DataFrame([input_data])
predicted_charge = gbm_model.predict(input_df)

# Display prediction
st.subheader("Predicted Insurance Charge")
st.write(predicted_charge[0])
