import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def detect_and_predict_mask(frame, face_cascade, maskNet):
    # Convert each frame to greyscale
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use Haar Cascade to detect faces within the frame
    faces = face_cascade.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Arrays for locations and predictions
    locs = []
    preds = []

    for (x, y, w, h) in faces:
        # Extracts the face region of the frame
        face = frame[y:y + h, x:x + w]
        # Converts face back to RGB and preprocesses image
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        # Appends location of face to location array
        locs.append((x, y, x + w, y + h))

        # Reshapes image array to shape expected by model
        # (batch size, height, width, channels)
        face = face.reshape((1, face.shape[0], face.shape[1], face.shape[2]))
        # Array is passed through the model and prediction results are appended to array
        preds.append(maskNet.predict(face))
    return locs, preds

def main():
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Load the face mask detector model
    maskNet = load_model("mask_detector.model")

    # Start the video capture, using system default camera
    vs = cv2.VideoCapture(0)

    while True:
        ret, frame = vs.read()

        # Detect faces and predict masks
        locs, preds = detect_and_predict_mask(frame, face_cascade, maskNet)

        # Loop over the detected face locations and their corresponding predictions
        for (box, pred) in zip(locs, preds):
            (left, top, right, bottom) = box
            (mask, withoutMask) = pred[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            # Colour is green if masked, red if not
            colour = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            # Display confidence
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Add the text and draw rectangle
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)

        # Display frame
        cv2.imshow("Frame", frame)
        
        # Quit application if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()