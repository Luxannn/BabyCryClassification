import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# WebRTC config
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.title("Baby Cry Predictor (Real-Time)")

st.subheader("Testing WebRTC Audio:")
st.write("Allow mic access. If it works, you’ll see a stream status below.")

# Basic WebRTC streamer to test audio
webrtc_ctx = webrtc_streamer(
    key="audio-test",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Check if stream is active
if webrtc_ctx.state.playing:
    st.write("Audio stream is active! Mic is working.")
else:
    st.write("Stream not active yet. Allow mic access and wait a sec.")
