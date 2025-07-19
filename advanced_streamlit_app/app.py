
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('output', exist_ok=True) 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import os

# Streamlit Page Setup
st.set_page_config(page_title="Advanced ML Deployment", layout="wide")

# Sidebar: Model Selection
st.sidebar.title('⚙️ Settings')
model_choice = st.sidebar.selectbox('Select Model', ['Random Forest', 'Logistic Regression', 'SVM'])

# Load Correct Model
if model_choice == 'Random Forest':
    model = pickle.load(open('model/random_forest_model.pkl', 'rb'))
elif model_choice == 'Logistic Regression':
    model = pickle.load(open('model/logistic_regression_model.pkl', 'rb'))
else:
    model = pickle.load(open('model/svm_model.pkl', 'rb'))

# App Title
st.title("🌸 Iris Flower Prediction (Advanced Version)")
st.write("Use sliders for input OR upload CSV for bulk predictions")

# Input Sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Manual Single Input Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
species = ['Setosa', 'Versicolor', 'Virginica']

st.write(f"### 🌼 Predicted Species: **{species[prediction]}**")

# Feature Bar Graph for Input
fig, ax = plt.subplots()
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
values = [sepal_length, sepal_width, petal_length, petal_width]
ax.barh(features, values, color='skyblue')
ax.set_xlabel('cm')
st.pyplot(fig)

# Automatically save feature input plot as image
fig.savefig('output/feature_input_plot.png')



import matplotlib.pyplot as plt
st.write("### 🔍 Why this prediction? (SHAP Visualization)")

background_data = pd.DataFrame(np.tile(input_data, (10, 1)), columns=features)
explainer = shap.Explainer(model, background_data)
shap_values = explainer(background_data)

# Convert shap_values to Explanation for class 0 properly
single_shap = shap.Explanation(
    values=shap_values.values[0, 0],
    base_values=shap_values.base_values[0, 0],
    data=shap_values.data[0],
    feature_names=shap_values.feature_names
)

fig, ax = plt.subplots(figsize=(6, 4))
shap.plots.waterfall(single_shap, max_display=4, show=False)
fig = plt.gcf()
st.pyplot(fig)


# Sidebar: CSV Upload for Bulk Prediction
st.sidebar.subheader("Or Upload CSV for Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.write(df)

    predictions = model.predict(df)
    df['Predicted Class'] = [species[i] for i in predictions]

    st.write("### 📊 Batch Predictions")
    st.write(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download predictions as CSV', data=csv, file_name='batch_predictions.csv', mime='text/csv')
    
    
    
    # Automatically save output CSV to output folder
    df.to_csv('output/batch_predictions.csv', index=False)

