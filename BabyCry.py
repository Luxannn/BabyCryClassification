import streamlit as st
import numpy as np
import pyaudio
import wave
import pickle
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Загрузите обученную модель
pickle_in = open("BabyCryModel.pkl","rb")
model=pickle.load(pickle_in)
#model = RandomForestClassifier(n_estimators=20, random_state=42)
#model.load_model("BabyCryModel.pkl")  # Замените "your_model_file.pkl" на реальное имя файла вашей модели

st.title("Baby Cry Predictor")

# Запись аудио с микрофона
st.subheader("Press 'Start' and make a baby cry sound:")
start_button = st.button("Start")

if start_button:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
    audio_frames = []
    st.text("Recording...")
    
    for _ in range(0, int(44100 / 1024 * 5)):  # Запись в течение 5 секунд
        audio_data = stream.read(1024)
        audio_frames.append(audio_data)
    
    st.text("Finished recording")
    p.terminate()
    
    # Сохранение записанного аудио в файл
    with wave.open("recorded_audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(audio_frames))
    
    # Загрузка и обработка записанного аудио, предсказание и отображение результатов
    try:
        audio_path = "recorded_audio.wav"
        
        with wave.open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.readframes(-1)
            sr = audio_file.getframerate()
            audio = np.frombuffer(audio_data, dtype=np.int16)
        
        # Преобразуйте данные в числа с плавающей точкой
        audio = audio.astype(np.float64)
        
        # Извлечем признаки из аудиофайла
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Сделайте прогноз с помощью модели
        prediction = model.predict([mfccs_mean])
        
        st.subheader("Prediction:")
        st.write(f"The baby's cry corresponds to: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during audio processing: {str(e)}")

