import streamlit as st
import time
import random

st.title("Пешгӯии Гиряи Кӯдак")

st.subheader("Тугмаи 'Оғоз' -ро пахш кунед ва садои гиряи кӯдакро барангезед:")
start_button = st.button("Оғоз")

if start_button:
    st.text("Сабт карда истодааст...")
    # Simulate 5-second "listening" like your original code
    time.sleep(5)
    st.text("Сабт анҷом ёфт")

    # Random prediction in Tajik to mimic your model's output
    try:
        cries = ["Гушна", "Хастагӣ", "Дард", "Малолат", "Тарс", "Хушҳол"]
        prediction = random.choice(cries)
        st.subheader("Пешгӯӣ:")
        st.write(f"Гиряи кӯдак ба ин мувофиқ аст: {prediction}")
    except Exception as e:
        st.error(f"Хато: {str(e)}")
