import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model("cnn_weights.h5")

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    '../data/test', target_size=(128,128), batch_size=32, class_mode='binary'
)

loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")
