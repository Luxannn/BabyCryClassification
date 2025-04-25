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

st.title("Baby Cry Predictor")

# WebRTC configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Audio processor for real-time recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 44100
        self.chunks_collected = 0
        self.max_chunks = int(44100 / 1024 * 5)  # Match your 5-second recording

    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float64)
        self.audio_buffer.append(audio)
        self.chunks_collected += 1

        # Process after ~5 seconds (match your original loop)
        if self.chunks_collected >= self.max_chunks:
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            self.chunks_collected = 0

            try:
                # Extract MFCC features (same as your code)
                mfccs = librosa.feature.mfcc(y=full_audio, sr=self.sample_rate)
                mfccs_mean = np.mean(mfccs, axis=1)

                # Predict (same as your code)
                prediction = model.predict([mfccs_mean])
                st.session_state["prediction"] = prediction[0]
            except Exception as e:
                st.session_state["prediction"] = f"Error: {str(e)}"

        return frame

st.subheader("Press 'Start' and make a baby cry sound:")
start_button = st.button("Start")

if start_button:
    st.text("Recording...")
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
    st.text("Finished recording")
    st.subheader("Prediction:")
    st.write(f"The baby's cry corresponds to: {st.session_state['prediction']}")
