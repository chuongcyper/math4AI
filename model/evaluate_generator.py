# Load the saved model
from keras.models import model_from_json

with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('emotion_model.h5')

# Compile the model
loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Evaluate the model on the test dataset
test_data_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

precision = loaded_model.evaluate_generator(test_generator, steps=len(test_generator))

print("Test Accuracy:", precision[1])
