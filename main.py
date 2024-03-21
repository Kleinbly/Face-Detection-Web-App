import cv2
import streamlit as st

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Face Detection using Viola-Jones Algorithm")
st.write("Press the 'Detect Faces' button to start detecting faces from your webcam.")

# Instructions for the user
st.write("Instructions:")
st.write("- Press the '**Detect Faces**' button to start detecting faces from your webcam.")
st.write("- Press the '**Stop**' button to stop the detection process.")
st.write("- Choose the **color** of the rectangles drawn around the detected faces using the color picker below.")
st.write("- Adjust the '**minNeighbors**' parameter to fine-tune face detection.")
st.write("- Adjust the '**scaleFactor**' parameter to fine-tune face detection.")

# Color picker for choosing the color of rectangles
color = st.color_picker("Choose the color of the rectangles", "#00ff00")

# Convert the hex color code to BGR format
bgr_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

# Slider to adjust the minNeighbors parameter
min_neighbors = st.empty()

# Slider to adjust the scaleFactor parameter
scale_factor = st.empty()

frame_placeholder = st.empty()

face_placeholder = st.empty()

stop_button_pressed = st.button("Stop")

save_button_pressed = st.button("Save face")


def detect_faces(scale_factor_value, min_neighbors_value):
    last_image = None
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    # Start capturing frames
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        if not ret:
            st.write("The video capture has ended")
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=float(scale_factor_value), minNeighbors=min_neighbors_value)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

            # Store the last detected face
            last_image = frame[y:y + h, x:x + w]
            face_placeholder.image(last_image, channels="GBR")


        frame_placeholder.image(frame, channels="GBR")



        # Check if 'Stop' button is clicked
        if stop_button_pressed:
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()


# Define the streamlit app
def app():
    min_neighbors_value = min_neighbors.slider("minNeighbors", 1, 10, 5)

    scale_factor_value =scale_factor.slider("scaleFactor", 1.1, 2.0, 1.2, 0.1)

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces(scale_factor_value, min_neighbors_value)


if __name__ == "__main__":
    # Initialize variable to store the last detected face image
    app()
