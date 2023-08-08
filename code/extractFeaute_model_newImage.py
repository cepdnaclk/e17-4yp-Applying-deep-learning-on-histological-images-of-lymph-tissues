import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model


def extract_and_save_feature_extractor(image_folder_path, save_path):
    # Load the pre-trained VGG16 model (excluding the classification layers)
    base_model = VGG16(weights='imagenet', include_top=False)

    # Remove the classification layers
    feature_extractor = Model(inputs=base_model.input,
                              outputs=base_model.layers[-1].output)

    # Get a list of image file names in the folder
    image_file_names = os.listdir(image_folder_path)

    # Load and preprocess images
    features_list = []
    for file_name in image_file_names:
        image_path = os.path.join(image_folder_path, file_name)
        # Resize to VGG16 input size
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        # Preprocess input according to VGG16 requirements
        preprocessed_img = preprocess_input(img_array)
        features = feature_extractor.predict(np.expand_dims(
            preprocessed_img, axis=0))  # Extract features
        # Flatten the features and store
        features_list.append(features.flatten())

    # Convert the list of extracted features to a NumPy array
    extracted_features = np.array(features_list)

    # Save the feature extractor model without classification layers
    feature_extractor.save(save_path)


# Define the path to your image folder
image_folder_path = os.path.abspath('../../MedNET/cancer images/images')

# Define the path to save the feature extractor model
save_path = os.path.abspath('../../MedNET/model/VGG16_feature_extractor.h5')

# Call the function to extract features from images and save the model
extract_and_save_feature_extractor(image_folder_path, save_path)
