import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = tf.keras.models.load_model('model.h5')

# --------------------------------------------------
# Load encoders and scaler
# --------------------------------------------------

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.title('Customer Churn Prediction')


# --------------------------------------------------
# User Input
# --------------------------------------------------

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

gender = st.selectbox(    'Gender',    label_encoder_gender.classes_)

age = st.slider('Age', 18,    92)

balance = st.number_input(    'Balance',    min_value=0.0)
credit_score = st.number_input(    'Credit Score',    min_value=0,    max_value=1000,    value=600)
estimated_salary = st.number_input(    'Estimated Salary',    min_value=0.0)
tenure = st.slider(    'Tenure',    0,    10)
num_of_products = st.slider(    'Number of Products',    1,    4)

has_cr_card = st.selectbox(    'Has Credit Card',    [0, 1])

is_active_member = st.selectbox(    'Is Active Member',    [0, 1])


# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# --------------------------------------------------
# One-Hot Encode Geography
# --------------------------------------------------

geo_encoded = onehot_encoder_geo.transform(    input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(    geo_encoded,    columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# --------------------------------------------------
# Encode Gender
# --------------------------------------------------

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])


# --------------------------------------------------
# Remove Geography column
# --------------------------------------------------

input_data = input_data.drop('Geography', axis=1)


# --------------------------------------------------
# Combine encoded Geography with input data
# --------------------------------------------------

input_data = pd.concat(    [ input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],axis=1)


# --------------------------------------------------
# Scale Input Data
# --------------------------------------------------

input_data_scaled = scaler.transform(input_data)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(input_data_scaled)

prediction_probability = prediction[0][0]


# --------------------------------------------------
# Display Prediction
# --------------------------------------------------

st.write(f'Churn Probability: {prediction_probability:.2%}')

if prediction_probability > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')