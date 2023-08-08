import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import (VGG16, ResNet50, MobileNet,
                                           InceptionV3, InceptionResNetV2,
                                           DenseNet121, DenseNet169, DenseNet201,
                                           EfficientNetB0, EfficientNetB1, EfficientNetB2,
                                           EfficientNetB3, EfficientNetB4, EfficientNetB5,
                                           EfficientNetB6, EfficientNetB7, SqueezeNet)
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.squeezenet import preprocess_input as squeezenet_preprocess_input
from tensorflow.keras.models import load_model


def extract_and_save_feature_extractor(image_folder_path, model_name, save_path):
    # Define the model and its preprocessing function based on the given model name
    model_dict = {
        'vgg16': (VGG16, vgg_preprocess_input),
        'resnet50': (ResNet50, resnet_preprocess_input),
        'mobilenet': (MobileNet, mobilenet_preprocess_input),
        'inceptionv3': (InceptionV3, inception_v3_preprocess_input),
        'inceptionresnetv2': (InceptionResNetV2, inception_resnet_v2_preprocess_input),
        'densenet121': (DenseNet121, densenet_preprocess_input),
        'densenet169': (DenseNet169, densenet_preprocess_input),
        'densenet201': (DenseNet201, densenet_preprocess_input),
        'efficientnetb0': (EfficientNetB0, efficientnet_preprocess_input),
        'efficientnetb1': (EfficientNetB1, efficientnet_preprocess_input),
        'efficientnetb2': (EfficientNetB2, efficientnet_preprocess_input),
        'efficientnetb3': (EfficientNetB3, efficientnet_preprocess_input),
        'efficientnetb4': (EfficientNetB4, efficientnet_preprocess_input),
        'efficientnetb5': (EfficientNetB5, efficientnet_preprocess_input),
        'efficientnetb6': (EfficientNetB6, efficientnet_preprocess_input),
        'efficientnetb7': (EfficientNetB7, efficientnet_preprocess_input),
        'squeezenet': (SqueezeNet, squeezenet_preprocess_input),
    }

    if model_name.lower() not in model_dict:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are: {', '.join(model_dict.keys())}")

    # Load the pre-trained model (excluding the classification layers)
    base_model = model_dict[model_name.lower()][0](
        weights='imagenet', include_top=False)

    # Remove the classification layers
    feature_extractor = Model(inputs=base_model.input,
                              outputs=base_model.layers[-1].output)

    # Get a list of image file names in the folder
    image_file_names = os.listdir(image_folder_path)

    # Load and preprocess images
    features_list = []
    for file_name in image_file_names:
        image_path = os.path.join(image_folder_path, file_name)
        # Resize to model's input size
        img = load_img(image_path, target_size=base_model.input_shape[1:3])
        img_array = img_to_array(img)
        # Preprocess input according to the model's requirements
        preprocessed_img = model_dict[model_name.lower()][1](img_array)
        features = feature_extractor.predict(np.expand_dims(
            preprocessed_img, axis=0))  # Extract features
        # Flatten the features and store
        features_list.append(features.flatten())

    # Convert the list of extracted features to a NumPy array
    extracted_features = np.array(features_list)

    # Save the feature extractor model without classification layers
    feature_extractor.save(save_path)


def test_feature_extraction(image_folder_path, save_dir):
    model_names = ['vgg16', 'resnet50', 'mobilenet', 'inceptionv3',
                   'inceptionresnetv2', 'densenet121', 'densenet169', 'densenet201',
                   'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
                   'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'squeezenet']

    for model_name in model_names:
        save_path = os.path.join(
            save_dir, f"{model_name}_feature_extractor.h5")
        extract_and_save_feature_extractor(
            image_folder_path, model_name, save_path)

        # Verify if the model is saved correctly
        assert os.path.exists(
            save_path), f"Failed to save the {model_name} feature extractor model."

        # Load the saved model to check if it is a valid Keras model
        try:
            loaded_model = load_model(save_path)
            assert isinstance(
                loaded_model, tf.keras.Model), f"The {model_name} saved model is not a valid Keras model."
        except Exception as e:
            assert False, f"Error while loading the {model_name} saved model: {e}"

        print(
            f"{model_name} feature extractor model saved successfully and passed validation.")


# Example usage:
image_folder_path = '/home/e17358/4yp/MedNET/medicalimages/'
save_dir = '/home/e17358/4yp/MedNET/MedNET/createdModels/'

test_feature_extraction(image_folder_path, save_dir)
