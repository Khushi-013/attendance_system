import cv2
import os
import streamlit as st
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from tensorflow import keras

from keras.models import load_model


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access Webcam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Emotion')

#### Get the total number of registered users
def total_reg():
    return len(os.listdir('static/faces'))

#### Extract the face from an image
def extract_faces(img):
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(face_array):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(face_array)

#### Detect emotion using ML model
def detect_emotion(face):
    model = load_model('emotion_detection_model.h5')
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_prediction = model.predict(face)[0]
    emotion = emotion_labels[np.argmax(emotion_prediction)]
    return emotion

#### A function to train the model on all the faces available in the faces folder
def train_model():
    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    for user in user_list:
        for img_name in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{img_name}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in the attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    emotions = df['Emotion']
    l = len(df)
    return names, rolls, times, emotions, l

#### Add attendance of a specific user
def add_attendance(name, emotion):
    username = name.split('_')[0]
    userid = name.split('_')[1] if len(name.split('_')) > 1 else ''
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if userid and int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{emotion}')


import time

# Rest of the code...

def clear_attendance():
    cap.release()  # Release webcam capture resources
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if os.path.isfile(attendance_file):
        os.remove(attendance_file)
        st.success("Attendance cleared successfully.")
    else:
        st.warning("No attendance file found.")



################## ROUTING FUNCTIONS #########################

#### Main page
def home():
    names, rolls, times, emotions, l = extract_attendance()
    st.title("Attendance System")
    st.write(f"Date: {datetoday2}")
    st.write(f"Total Registered Users: {total_reg()}")

    if st.button("Take Attendance"):
        st.write("Taking attendance...")
        start()

    if st.button("Clear Attendance"):
        clear_attendance()

    st.write("Attendance:")
    attendance_df = pd.DataFrame({"Name": names, "Roll": rolls, "Time": times, "Emotion": emotions})
    st.write(attendance_df)

#### Run when clicking on Take Attendance button
import streamlit as st

def start():
    stop_camera = False  # Variable to control stopping the camera
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        st.warning("There is no trained model in the static folder. Please add a new face to continue.")
        return

    while not stop_camera:
        ret, frame = cap.read()
        if ret:
            face_points = extract_faces(frame)
            if len(face_points) > 0:
                (x, y, w, h) = face_points[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]

                # Perform emotion detection on the detected face
                emotion = detect_emotion(face)

                add_attendance(identified_person, emotion)

                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f'Emotion: {emotion}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)

            # Display the frame using Streamlit's image function
            st.image(frame, channels="BGR", use_column_width=True)

        # Sleep for a short interval to capture frames asynchronously
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, emotions, l = extract_attendance()
    attendance_df = pd.DataFrame({"Name": names, "Roll": rolls, "Time": times, "Emotion": emotions})
    st.write("Attendance:")
    st.write(attendance_df)




def add():
    new_username = st.text_input("New User Name:")
    new_userid = st.text_input("New User ID:")
    if st.button("Add User"):
        user_image_folder = 'static/faces/' + new_username + '_' + str(new_userid)
        if not os.path.isdir(user_image_folder):
            os.makedirs(user_image_folder)
        i, j = 0, 0
        stop_camera = False  # Variable to control stopping the camera

        # Generate a unique key for the "Stop" button using the user name and user ID
        stop_button_key = f"stop_button_{new_username}_{new_userid}"

        # Add the "Stop" button outside the loop
        stop_button = st.button("Stop", key=stop_button_key)

        while j < 500 and not stop_camera:
            ret, frame = cap.read()
            if ret:
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                    if j % 10 == 0:
                        name = new_username + '_' + str(i) + '.jpg'
                        cv2.imwrite(user_image_folder + '/' + name, frame[y:y + h, x:x + w])
                        i += 1
                    j += 1
                # Display the frame using Streamlit's image function
                st.image(frame, channels="BGR", use_column_width=True)

            # Check if the "Stop" button is clicked
            if stop_button:
                stop_camera = True

        cv2.destroyAllWindows()
        st.write("Training Model...")
        train_model()
        names, rolls, times, emotions, l = extract_attendance()
        attendance_df = pd.DataFrame({"Name": names, "Roll": rolls, "Time": times, "Emotion": emotions})
        st.write("Attendance:")
        st.write(attendance_df)

# ...





# ...



# ...


#### Run when selecting a user
def select_user():
    user_list = os.listdir('static/faces')
    selected_user = st.selectbox("Select User:", user_list)
    user_images = os.listdir(f'static/faces/{selected_user}')
    if len(user_images) > 0:
        st.write(f"Number of Images: {len(user_images)}")
        for image_name in user_images:
            image_path = f'static/faces/{selected_user}/{image_name}'
            st.image(image_path, use_column_width=True)
    else:
        st.warning("No images found for the selected user.")

#### Main function to run the Streamlit App
def main():
    st.set_page_config(page_title="Attendance Tracking System Using Facial Technology")
    menu = ["Home", "Add User", "View Registered Users"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        home()
    elif choice == "Add User":
        add()
    elif choice == "View Registered Users":
        select_user()

if __name__ == '__main__':
    main()
