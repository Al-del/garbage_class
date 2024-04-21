import cv2
import numpy as np
import tensorflow as tf
# Open the video stream
cap = cv2.VideoCapture(0)  # 0 for default camera, or replace with video file path
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
model = tf.keras.models.load_model('my_keras_model.h5')
print(model.summary())
while True:
    # Read frame by frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to size 224, 224, 3
    resized_frame = cv2.resize(frame, (224, 224))

    # Add an extra dimension to represent the batch size
    resized_frame = np.expand_dims(resized_frame, axis=0)
 
    # Fit the resized frame into the model
    prediction = model.predict(resized_frame)
    #Print the class predicted
    print(class_names[prediction.argmax()])
    cv2.imshow('Frame', frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()