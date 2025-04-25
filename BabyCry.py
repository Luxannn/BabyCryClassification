import streamlit as st
import numpy as np
import pickle
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import av

# Load the trained model
try:
    pickle_in = open("BabyCryModel.pkl", "rb")
    model = pickle.load(pickle_in)
except FileNotFoundError:
    st.error("Model file 'BabyCryModel.pkl' not found. Please upload it to the repo.")
    st.stop()

# WebRTC configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Audio processor for real-time prediction
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 44100  # Match your original 44100 Hz

    def recv(self, frame):
        # Convert audio frame to numpy array
        audio = frame.to_ndarray().flatten().astype(np.float64)
        self.audio_buffer.append(audio)

        # Check if we have ~5 seconds of audio
        if len(self.audio_buffer) * self.sample_rate / 1000 > 5:
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []  # Reset buffer

            try:
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=full_audio, sr=self.sample_rate)
                mfccs_mean = np.mean(mfccs, axis=1)

                # Predict with model
                prediction = model.predict([mfccs_mean])
                st.session_state["prediction"] = prediction[0]
            except Exception as e:
                st.session_state["prediction"] = f"Error: {str(e)}"

        return frame

st.title("Baby Cry Predictor (Real-Time)")

st.subheader("Allow microphone access to listen for baby cries:")
st.write("This app will predict after ~5 seconds of audio.")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Display prediction
if "prediction" in st.session_state:
    st.subheader("Prediction:")
    st.write(f"The baby's cry corresponds to: {st.session_state['prediction']}")
