from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
EPOCHS = 10
BATCH_SIZE = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224
TEST_SIZE = 0.2

# Dataset directory information
DIRECTORY = "archive"
CATEGORIES = ["with_mask 1/with_mask 1", "without_mask/without_mask"]

def load_data():
    data = []  # For loaded image data (in array form)
    labels = []  # For classes of images

    # Image preprocessing
    try:
        # Loop through each category of images
        for category_name in CATEGORIES:
            category_path = os.path.join(DIRECTORY, category_name)
            # Iterate through each image in the category
            for image_file in os.listdir(category_path):
                try:
                    image_array = img_to_array(load_img(
                                                os.path.join(category_path, image_file),
                                                target_size=(IMG_WIDTH, IMG_HEIGHT)
                                                ))
                    preprocessed_image = preprocess_input(image_array)
                    data.append(preprocessed_image)
                    labels.append(category_name)
                except Exception as e:
                    print(f"Error processing image {image_array}: {e}")
    except Exception as e:
        print(f"Error accessing dataset: {e}")

    # One hot encoding on labels
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    labels = to_categorical(labels)

    # Convert array to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # Return data and labels
    return(data, labels)

def get_model():
    # Construct the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_plot(history):
    # Plot the training loss and accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epoch')
    plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='lower left')
    plt.savefig("plot.png")
    
def generate_report(model, testX, testY):
    # Predict the class probabilities for test data
    predicted_probabilities = model.predict(testX, batch_size=BATCH_SIZE)

    # Convert predicted probabilities into class labels
    predicted_classes = np.argmax(predicted_probabilities, axis=1)

    # Prepare the true class labels for the test data by converting them back from one-hot encoded format
    true_classes = np.argmax(testY, axis=1)

    # Generate a classification report comparing true classes with predicted classes
    report = classification_report(true_classes, predicted_classes, target_names=CATEGORIES)
    report_file_path = 'report.txt'
    with open(report_file_path, 'w') as report_file:
        report_file.write(report)

def main():
    (data, labels) = load_data()
    
    # Training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=18,
        zoom_range=0.2,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.12,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Create train test split
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=1)

    model = get_model()
    
    # Train the model
    history = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Create plot and generate report
    create_plot(history)    
    generate_report(model, testX, testY)

    # Save model
    model.save("mask_detector.model", save_format="h5")
    
if __name__ == "__main__":
    main()