import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import build_cnn

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    '../data/train', target_size=(128,128), batch_size=32, class_mode='binary'
)

# Build and train model
model = build_cnn()
model.fit(train_data, epochs=5)
model.save("cnn_weights.h5")
