import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = './Crop_recommendation.csv'
df = pd.read_csv(file_path)

# Features and target variable
X = df.drop('label', axis=1)
y = df['label']

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

### Streamlit app ###
st.title('🌾 Crop Recommendation System')

st.markdown("""
Welcome to the Crop Recommendation System! This tool helps you identify the most suitable crop to grow based on several soil and weather parameters.
Please provide the necessary inputs in the sidebar to get a crop recommendation.
""")

st.sidebar.header('Input Parameters')
st.sidebar.markdown("""
Adjust the parameters below to get a crop recommendation.
""")

def user_input_features():
    N = st.sidebar.slider('Nitrogen (N)', int(X['N'].min()), int(X['N'].max()), int(X['N'].mean()))
    P = st.sidebar.slider('Phosphorus (P)', int(X['P'].min()), int(X['P'].max()), int(X['P'].mean()))
    K = st.sidebar.slider('Potassium (K)', int(X['K'].min()), int(X['K'].max()), int(X['K'].mean()))
    temperature = st.sidebar.slider('Temperature (°C)', float(X['temperature'].min()), float(X['temperature'].max()), float(X['temperature'].mean()))
    humidity = st.sidebar.slider('Humidity (%)', float(X['humidity'].min()), float(X['humidity'].max()), float(X['humidity'].mean()))
    ph = st.sidebar.slider('pH', float(X['ph'].min()), float(X['ph'].max()), float(X['ph'].mean()))
    rainfall = st.sidebar.slider('Rainfall (mm)', float(X['rainfall'].min()), float(X['rainfall'].max()), float(X['rainfall'].mean()))
    
    data = {'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Parameters')
# Convert DataFrame to HTML and hide the index using CSS
st.markdown(input_df.to_html(index=False), unsafe_allow_html=True)

# Make the crop predictions
prediction = rf.predict(input_df)
prediction_proba = rf.predict_proba(input_df)

# Add a bit of space between each area of the page
st.markdown("""<br>""", unsafe_allow_html=True)

st.subheader('Crop Recommendation')
st.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center; padding: 10px; border-radius: 5px; background-color: #f5f5f5; border: 1px solid #ddd;">
    <h3 style="color: #4CAF50; margin: 0;"><strong>{prediction[0].upper()}</strong></h3>
</div>
""", unsafe_allow_html=True)

# Add a bit of space between each area of the page
st.markdown("""<br>""", unsafe_allow_html=True)

st.subheader('Prediction Probabilities')
st.markdown("""
The bar chart below shows the probabilities for each crop based on the input parameters.
A higher probability indicates a higher likelihood of being the most suitable crop to grow.
""")
prob_df = pd.DataFrame(prediction_proba, columns=rf.classes_)
prob_df_transposed = prob_df.T.reset_index()
prob_df_transposed.columns = ['Crop', 'Probability']

st.bar_chart(prob_df_transposed.set_index('Crop'))

# Add a footer with a copyright claim
st.markdown("""
<br>
<br>
<hr>
<p style="text-align: center;">&copy; 2024 Cuckoo Lab. All rights reserved.</p>
""", unsafe_allow_html=True)
