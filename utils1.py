from ultralytics import YOLO
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import cv2
from PIL import Image
import tempfile
import config
import threading

def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)
    
    inText = 'Incoming'
    outText = 'Outgoing'
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += ' - ' + str(key) + ": " +str(value)
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += ' - ' + str(key) + ": " +str(value)
    
    # Plot the detected objects on the video frame
    st_count.write(inText + '\n\n' + outText)
    print(inText + '\n\n' + outText)
    
    res_plotted = res[0].plot()
    
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                
                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                        personcount = config.person_count.item()
                        st.text("People Count: " + str(personcount))

                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)

def process_video(conf, model, st_count, st_frame, source_video):
    try:
        config.OBJECT_COUNTER1 = None
        config.OBJECT_COUNTER = None
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(source_video.read())
        vid_cap = cv2.VideoCapture(tfile.name)
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_count, st_frame, image)
            else:
                cv2.waitKey(3000)
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {e}")

def infer_uploaded_video(conf, model):

    col1, col2 = st.columns(2)
    
    with col1:
        source_video1 = st.sidebar.file_uploader(label="Choose a video1...")
    
        if source_video1:
            st.video(source_video1)
            if st.button("Monitor Cam1"):
                with st.spinner("Running Cam1..."):
                    try:
                        st_count1 = st.empty()
                        st_frame1 = st.empty()
                        thread = threading.Thread(target=process_video, args=(conf, model, st_count1, st_frame1, source_video1))
                        add_script_run_ctx(thread)
                        thread.start()
                    except Exception as e:
                        st.error(f"Error loading video: {e}")

                    #personincount = config.person_in
                    #st.text("People Entering Count: " + str(personincount))
                    #personoutcount = config.person_out
                    #st.text("People Leaving Count: " + str(personoutcount))
                    
                    
    with col2:
        source_video2 = st.sidebar.file_uploader(label="Choose a video2...")
        if source_video2:
            st.video(source_video2)
            if st.button("Monitor Cam2"):
                with st.spinner("Running Cam2..."):
                    try:
                        st_count2 = st.empty()
                        st_frame2 = st.empty()
                        thread = threading.Thread(target=process_video, args=(conf, model, st_count2, st_frame2, source_video2))
                        add_script_run_ctx(thread)
                        thread.start()
                    except Exception as e:
                        st.error(f"Error loading video: {e}")

                    #personincount = config.person_in
                    #st.text("People Entering Count: " + str(personincount))
                    #personoutcount = config.person_out
                    #st.text("People Leaving Count: " + str(personoutcount))

def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
