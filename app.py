import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
import pickle
import sys

model = tf.keras.models.load_model('model.h5')
encode = pickle.load(open('encode.pkl','rb'))
scale = pickle.load(open('scale.pkl','rb'))


st.title("Costomer Churn Prediction")

# user Input
Age = st.slider('Age',18,92)
Gender	= st.selectbox("Gender",['Female','Male'])
Tenure = st.slider("Tenure",0,10)
Usage_Frequency	= st.slider('Frequency of use',0,60)
Support_Calls	= st.slider('Support Calls',0,20)
Payment_Delay	= st.slider('Payment Delay',0,60)
Total_Spend	= st.number_input('Totel spend',1000)
Last_Interaction = st.number_input('Last Interaction',10)

Input_data = pd.DataFrame({
    'Age':[Age],
    'Gender':[encode.transform([Gender])[0]],
    'Tenure':[Tenure],
    "Usage Frequency":[Usage_Frequency],
    'Support Calls':[Support_Calls],
    'Payment Delay':[Payment_Delay],
    'Total Spend': [ Total_Spend ],
    'Last Interaction':[Last_Interaction]
})


input_data_scale = scale.transform(Input_data)

predict = model.predict(input_data_scale)

if predict[0][0] > 0.5:
        st.write("The Customer is likely to churn",predict)
else:
        st.write("The Customer is not likely to churn")