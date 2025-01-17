import streamlit as st 
from home import utils
from streamlit_webrtc import webrtc_streamer
import av
import time

# st.set_page_config(page_title='Predictions')
st.subheader('Real-Time Attendance System')
print("before")


# Retrive the data from Redis Database
import streamlit as st
import utils

# Use spinner for data retrieval
with st.spinner('Retrieving Data from Redis DB...'):
    try:
        redis_face_db = utils.retrieve_data(name='academy:register')
        if redis_face_db is not None:
            st.dataframe(redis_face_db)
        else:
            st.error("Failed to retrieve data from Redis.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

print("after")

    
st.success("Data sucessfully retrived from Redis")

# time 
waitTime = 30 # time in sec
setTime = time.time()
realtimepred = utils.RealTimePred() # real time prediction class

# Real Time Prediction
# streamlit webrtc
# callback function
def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") # 3 dimension numpy array
    # operation that you can perform on the array
    pred_img = realtimepred.face_prediction(img,redis_face_db,
                                        'facial_features',['Name','Role'],thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() # reset time        
        print('Save Data to redis database')
    

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)