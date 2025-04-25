import streamlit as st
import time
import random

st.title("Baby Cry Predictor")

st.subheader("Press 'Start' and make a baby cry sound:")
start_button = st.button("Start")

if start_button:
    st.text("Recording... (pretending to listen for 5 seconds)")
    # Simulate 5-second "listening" like your original code
    time.sleep(5)
    st.text("Finished recording")

    # Random prediction to mimic your model's output
    try:
        cries = ["Hungry", "Tired", "Pain", "Bored", "Scared", "Happy"]
        prediction = random.choice(cries)
        st.subheader("Prediction:")
        st.write(f"The baby's cry corresponds to: {prediction}")
    except Exception as e:
        st.error(f"Oops, something broke: {str(e)}")
