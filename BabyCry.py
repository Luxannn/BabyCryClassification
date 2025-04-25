import streamlit as st
import numpy as np
import pickle
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
import av

# Load the trained model
try:
    pickle_in = open("BabyCryModel.pkl", "rb")
    model = pickle.load(pickle_in)
except FileNotFoundError:
    st.error("Model file 'BabyCryModel.pkl' not found. Please upload it to the repo.")
    st.stop()

# WebRTC configuration (optional, improves connection)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Audio processor class to handle real-time audio
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame):
        # Convert audio frame to numpy array
        audio = frame.to_ndarray().flatten().astype(np.float64)
        self.audio_buffer.append(audio)

        # Process 5 seconds of audio (adjust based on your needs)
        if len(self.audio_buffer) * frame.sample_rate / 1000 > 5:  # Roughly 5 seconds
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []  # Reset buffer

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=full_audio, sr=frame.sample_rate)
            mfccs_mean = np.mean(mfccs, axis=1)

            # Predict with the model
            prediction = model.predict([mfccs_mean])
            st.session_state["prediction"] = prediction[0]

        return frame

st.title("Baby Cry Predictor (Real-Time)")

# Instructions
st.subheader("Allow microphone access and start listening for baby cries:")
st.write("This app will listen in real-time and predict after 5 seconds of audio.")

# WebRTC streamer for real-time audio
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode="sendonly",
    audio_processor_factory=AudioProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Display prediction
if "prediction" in st.session_state:
    st.subheader("Prediction:")
    st.write(f"The baby's cry corresponds to: {st.session_state['prediction']}")
