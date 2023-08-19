import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import os
import subprocess




loaded_model = load_model("C:/Users/VISHAL KHUMAR P D/Downloads/Final_model.h5")


IMAGE_HEIGHT , IMAGE_WIDTH = 100,100

CLASSES_LIST =  [
                    "Basketball", "BoxingPunchingBag", "GolfSwing", "IceDancing", "JugglingBalls",
                    "HorseRace", "LongJump", "Shotput", "TennisSwing", "PushUps"
               ]


SEQUENCE_LENGTH = 20


def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
 

    video_reader = cv2.VideoCapture(video_file_path)

   
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

  
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

   
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

   
    predicted_class_name = ''

    
    while video_reader.isOpened():

       
        ok, frame = video_reader.read() 
        
       
        if not ok:
            break

    
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
       
        normalized_frame = resized_frame / 255

        
        frames_queue.append(normalized_frame)

  
        if len(frames_queue) == SEQUENCE_LENGTH:


            predicted_labels_probabilities = loaded_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            predicted_label = np.argmax(predicted_labels_probabilities)

            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(frame)
    video_reader.release()
    video_writer.release()

  
def main():  
    # giving a title
    st.title('Sports action Classification-Web App')
    #Upload video file
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        #store the uploaded video locally
        with open(os.path.join("C:/Users/VISHAL KHUMAR P D/Desktop/VideoClassificationApp-main/Temp/",uploaded_file.name.split("/")[-1]),"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")

        if st.button('Apply Deep learning classification'):
            output_file_path = os.path.join("C:/Users/VISHAL KHUMAR P D/Desktop/VideoClassificationApp-main/VideoSample/",
                                            uploaded_file.name.split("/")[-1].split(".")[0] + "_output1.mp4")
            with st.spinner('Please wait...Your video is processing.'):
                predict_on_video("C:/Users/VISHAL KHUMAR P D/Desktop/VideoClassificationApp-main/Temp/" +
                                 uploaded_file.name.split("/")[-1], output_file_path, SEQUENCE_LENGTH)

                #OpenCV’s mp4v codec is not supported by HTML5 Video Player at the moment, one just need to use another encoding option which is x264 in this case 
                os.chdir('C://Users/VISHAL KHUMAR P D/Desktop/VideoClassificationApp-main/VideoSample/')
                subprocess.call(['ffmpeg','-y', '-i', uploaded_file.name.split("/")[-1].split(".")[0]+".mp4",'-vcodec','libx264','-f','mp4','output4.mp4'],shell=True)
                st.success('Done!')

                video_path = os.path.join("C:/Users/VISHAL KHUMAR P D/Desktop/VideoClassificationApp-main/VideoSample/" + uploaded_file.name.split("/")[-1].split(".")[0]+"_output1.mp4")
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
    
    else:
        st.text("Please upload a video file")
    
    
    
if __name__ == '__main__':
    main()
