
import face_recognition
import cv2
import time
import numpy as np

# Converts images to 8bit RGB-format
def convert_img(bgr_im):
    rgb_im = cv2.cvtColor(bgr_im,cv2.COLOR_BGR2RGB)
    rgb_im.astype('uint8')
    return rgb_im

# Load images with faces and create facial feature encodings

# Image 1. Sander
sanderimg = cv2.imread("sander.jpg")
rgb_sanderim = convert_img(sanderimg)

# Image 2. Paulina
#pau = cv2.imread("pau.jpg")
#rgb_pau = convert_img(pau)

#print(rgb_pau.dtype)

#pau_face_encoding = face_recognition.face_encodings(rgb_pau)[0]
sander_face_encoding = face_recognition.face_encodings(rgb_sanderim)[0]

# Facial feature encodings and names are stored in lists
encodings_known_faces = [sander_face_encoding]
known_faces_names = ['Sander']

# Init video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Take a still frame (cv2 uses BGR) and convert to RGB so that face_recognition can process it    
    ret, frame = video_capture.read()
    rgb_frame = convert_img(frame)

    print("Video Capture")
    cv2.imshow('Video', frame)

    print(rgb_frame.dtype)
    print(rgb_frame.shape)

    if rgb_frame is not None and len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        # Proceed
    else:
        print("Frame is not valid for face recognition.")

    # Loops through x- and y-axis of the images of located faces that is also paires (zip) with the correspending encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # Compare the images
        matches = face_recognition.compare_faces(encodings_known_faces, face_encoding)
        name = "Unknown"

        # Get euclidian distances from the encodings of known faces to found faces  
        face_distances = face_recognition.face_distance(encodings_known_faces, face_encoding)
        # Best match is smallest euclidean distance
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # Draw box around the faces that were found
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Add label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

