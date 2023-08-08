import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, Model

def extract_features_and_save_model(base_model, image_folder_path, save_features_path, save_model_path):
    # Get a list of image file names in the folder
    image_file_names = os.listdir(image_folder_path)
    
    # Load and preprocess images
    features_list = []
    for file_name in image_file_names:
        image_path = os.path.join(image_folder_path, file_name)
        img = load_img(image_path, target_size=(224, 224))  # Resize to match the input size of the model
        img_array = img_to_array(img)
        preprocessed_img = preprocess_input(img_array)  # Preprocess input according to the model's requirements
        features = base_model.predict(np.expand_dims(preprocessed_img, axis=0))  # Extract features
        features_list.append(features.flatten())  # Flatten the features and store
        
    # Convert the list of extracted features to a NumPy array
    extracted_features = np.array(features_list)
    
    # Save the new model without classification layers
    new_model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    new_model.save(save_model_path)

# Define the path to your image folder
image_folder_path = os.path.abspath('../../MedNET/cancer images/images')

# Define the path to load the pre-trained model
model_path = 'path/to/your/model.h5'

# Load the pre-trained model
base_model = load_model(model_path)

# Define the path to save the extracted features
save_features_path = 'path/to/save/extracted_features.npy'

# Define the path to save the new model without classification layers
save_model_path = 'path/to/save/new_model_without_classification.h5'

# Call the function to extract features and save the model
extract_features_and_save_model(base_model, image_folder_path, save_features_path, save_model_path)
