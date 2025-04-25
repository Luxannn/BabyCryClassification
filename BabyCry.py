import streamlit as st
import random

st.title("Baby Cry Predictor (Random AF)")

st.subheader("Hit the button for a wild guess:")
start_button = st.button("Predict Some Shit")

if start_button:
    # Fake predictions, just random crap
    cries = ["Hungry", "Tired", "Pain", "Bored", "Scared", "Happy"]
    prediction = random.choice(cries)
    st.text("Thinking real hard...")
    st.subheader("Prediction:")
    st.write(f"This baby’s probably: {prediction}")
