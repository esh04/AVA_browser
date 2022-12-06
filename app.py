import streamlit as st
import pandas as pd
import numpy as np
from streamlit_player import st_player
import cv2 as cv

st.title('AVA Speaker Data Browser')

with open('data/videos.txt') as f:
    videos = f.readlines()

# remove whitespace characters like `\n` at the end of each line
videos = [x.strip() for x in videos]
video_name = videos[0]

video_name = st.sidebar.selectbox('Select a video', videos)

# load corresponding video from link
video_link = 'https://s3.amazonaws.com/ava-dataset/trainval/' + video_name


# load corresponding csv file from either folders
csv_name = video_name.split('.')[0] + '-activespeaker.csv'

try: 
    df = pd.read_csv('data/ava_activespeaker_test_v1.0/' + csv_name)
except:
    df = pd.read_csv('data/ava_activespeaker_train_v1.0/' + csv_name)

# df.columns = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2',
#                 'entity_box_y2','label','entity_id', 'spkid']

df.columns = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2',
                'entity_box_y2','label','entity_id']

# st.write(df)

# make a rectangle around the speaker
def draw_rect(img, x1, y1, x2, y2):
    x1 = int(x1*img.shape[1])
    y1 = int(y1*img.shape[0])
    x2 = int(x2*img.shape[1])
    y2 = int(y2*img.shape[0])
    img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

# get the image from the video of the given timestamp
def get_image(video_link, timestamp):
    cap = cv.VideoCapture(video_link)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.set(cv.CAP_PROP_POS_MSEC, timestamp*1000)
    ret, frame = cap.read()
    return frame

# have two views 
# 1. depending on video timestamp show info
# 2. video with speaker rectangle

# make navigation bar
nav = st.sidebar.selectbox('Select a view', ['Video Analysis', 'Speaker Analysis'])

if(nav == 'Video Analysis'):
    # show video
    play_back = st_player(video_link)
    print(play_back)

    # show info
    st.write(df)

else:
    # choose a timestamp
    num_timestamp = len(df)
    timestamp_idx = st.sidebar.slider('Select a timestamp', min_value=0, max_value=num_timestamp, step=1)

    # get the image from the video of the given timestamp
    img = get_image(video_link,  df['frame_timestamp'][timestamp_idx])
    img = draw_rect(img, df['entity_box_x1'][timestamp_idx], df['entity_box_y1'][timestamp_idx], df['entity_box_x2'][timestamp_idx], df['entity_box_y2'][timestamp_idx])


    # display image
    st.image(img, use_column_width=True)
    # add label to image 
    st.write(df['label'][timestamp_idx] + ' ' + str(df['entity_id'][timestamp_idx]))

