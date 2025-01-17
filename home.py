import av
from streamlit_webrtc import webrtc_streamer

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="key", video_frame_callback=video_frame_callback)
