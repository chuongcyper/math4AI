# import required packages
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Function to install scipy if not already installed
def install_scipy_if_needed():
    try:
        import scipy
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy'])

# Install scipy if needed
install_scipy_if_needed()

# Initialize image data generator with rescaling and data augmentation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all train images with data augmentation
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Preprocess all validation images without data augmentation
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(BatchNormalization())
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# Set up the ExponentialDecay schedule for learning rate decay
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.96
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Use the updated optimizer with learning rate schedule
optimizer = Adam(learning_rate=lr_schedule)
emotion_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the neural network/model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64,
    callbacks=[early_stopping]
)

# Save model structure in JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights in .h5 file
emotion_model.save_weights('emotion_model.h5')

# Display the final training loss and accuracy
print("Final Training Loss:", emotion_model_info.history['loss'][-1])
print("Final Training Accuracy:", emotion_model_info.history['accuracy'][-1])
print("Final Validation Loss:", emotion_model_info.history['val_loss'][-1])
print("Final Validation Accuracy:", emotion_model_info.history['val_accuracy'][-1])
