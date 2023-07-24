import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False  # Set shuffle to False to get correct predictions corresponding to classes.
)

# Do prediction on test data
predictions = emotion_model.predict(test_generator)

# Get the predicted class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get the true class labels
true_classes = test_generator.classes

# Calculate the confusion matrix
c_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(c_matrix)

# Calculate accuracy
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=emotion_dict.values()))

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict.values())
cm_display.plot(cmap=plt.cm.Blues)
plt.show()
