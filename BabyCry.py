import streamlit as st
import numpy as np
import pickle
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
try:
    pickle_in = open("BabyCryModel.pkl", "rb")
    model = pickle.load(pickle_in)
except FileNotFoundError:
    st.error("Model file 'BabyCryModel.pkl' not found. Please upload it to the repo.")
    st.stop()

st.title("Baby Cry Predictor")

# File upload for audio
st.subheader("Upload a WAV file to predict the baby's cry:")
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        # Load audio directly with librosa
        audio, sr = librosa.load(uploaded_file, sr=44100)
        st.text("Audio loaded successfully!")

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Make prediction with the model
        prediction = model.predict([mfccs_mean])

        # Display results
        st.subheader("Prediction:")
        st.write(f"The baby's cry corresponds to: {prediction[0]}")

    except Exception as e:
        st.error(f"Error during audio processing: {str(e)}")
