from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import io
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import PIL
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
UPLOAD_FOLDER = 'static/uploads'

model = keras.models.load_model("NEW_HER2Sample_VGG16.h5")


def transform_image(file_path):
    # Open the saved image file
    pillow_image = Image.open(file_path)

    # Convert grayscale image to RGB if needed
    if pillow_image.mode != 'RGB':
        pillow_image = pillow_image.convert('RGB')

    data = np.asarray(pillow_image)
    data = data[np.newaxis, ...]
    data = tf.image.resize(data, [224, 224])
    return data


def predict(x):
    predictions = model.predict(x)
    print('predictions', predictions)
    predictions = tf.nn.softmax(predictions)
    print('predictions', predictions)

    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def linePredict(saved_image_path, class_index_to_explain=1):
    # Create a Lime explainer for image classification
    explainer = lime_image.LimeImageExplainer()

    # Load the saved image for explanation
    image = PIL.Image.open(saved_image_path)

    # Resize the image to the expected input shape (224, 224)
    image = image.resize((224, 224))

    # Preprocess the image for the vgg16 model
    preprocessed_image = tf.keras.applications.vgg16.preprocess_input(
        np.array(image))

    # Predict the class probabilities for the image using your model
    class_probabilities = model.predict(
        np.expand_dims(preprocessed_image, axis=0))

    print("check point 1")
    # Generate an explanation for the prediction
    explanation = explainer.explain_instance(
        preprocessed_image, model.predict, top_labels=2, num_samples=50)

    # Get the explanation for the class of interest
    explanation_for_class = explanation.top_labels[class_index_to_explain]
    print("check point 2")

    # Visualize the explanation
    try:
        explanation_image, mask = explanation.get_image_and_mask(
            class_probabilities.argmax(axis=1)[0], positive_only=False, num_features=4, hide_rest=False)
    except Exception as e:
        print("Error", e)
        return  # Handle the error or return an appropriate response

    # Normalize the explanation image to the range [0, 1]
    explanation_image = (explanation_image - explanation_image.min()) / (
        explanation_image.max() - explanation_image.min())

    print("check point 3")

    # Save the explanation image
    explanation_image_path = os.path.join(
        app.config['UPLOAD_FOLDER'], "explanation_lime.png")
    print(explanation_image_path)
    plt.imsave(explanation_image_path, mark_boundaries(
        explanation_image / 2 + 0.5, mask))

    return explanation_image_path

def generate_shap_mask1(image_path):
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # dataGenerator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,validation_split=0.1)
    
    # train_batches = dataGenerator.flow_from_directory(directory=image_path, target_size=(224, 224), classes=['0','1+','2+','3+'], batch_size=32,class_mode='categorical',subset='training')

    # # Create a DeepExplainer instance for your model
    # background = train_batches[0][0]
    e = shap.DeepExplainer(model)

    # Load the test image
    image = np.array(Image.open(image_path).convert("RGB"))
    image = image / 255.0  # Normalize to [0, 1]

    # Compute Shapley values for the image
    shap_values = e.shap_values(np.expand_dims(image, axis=0))

    # # Display the image
    # plt.imshow(image)
    # plt.title("Original Image")
    # plt.axis('off')
    # plt.show()

    # # Visualize the Shapley values for this image
    # shap.image_plot(shap_values[0][0], -image)  # Assuming shap_values[i] is a list with one element
    # plt.title("Shapley Values")
    # plt.axis('off')
    # plt.show()

    # Save the Shapley values image
    explanation_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "explanation_shap.png")
    plt.savefig(explanation_image_path, bbox_inches='tight', pad_inches=0)
    return explanation_image_path

def f(X):
    tmp = X.copy()
    preprocess_input(tmp)
    return model(tmp)

def generate_shap_mask(image_path):
    # Load the test image
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img.convert("RGB")).reshape(1, 224, 224, 3)
    # img_array = preprocess_input(img_array)

    # Define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("inpaint_telea", img_array[0].shape)

    # By default, the Partition explainer is used for all partition explainer
    explainer = shap.Explainer(f, masker, output_names=['0','1+','2+','3+'])

    # Here we use 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(img_array, max_evals=500, batch_size=100, outputs=shap.Explanation.argsort.flip[:1])
# shap.image_plot(shap_values[i][0], -image)
    # Visualize the SHAP values
    shap.image_plot(shap_values , -img_array,show= False)

    # Save the Shapley values image
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], "explanation_shap.png")

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return save_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return render_template("index.html", error="No file selected")

        try:
            filename = secure_filename(file.filename)
            unique_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
            file.save(unique_filename)

            # Display the selected image
            selected_image_path = f"/{unique_filename}"
            tensor = transform_image(unique_filename)
            prediction = predict(tensor)
            data = {"prediction": int(prediction)}
            print('Result:', data)
            try:
                limePath = linePredict(unique_filename)
            except Exception as e:
                print('Error lime:', e)
            try:
                shapPath = generate_shap_mask(unique_filename)
            except Exception as e:
                print('Error shap:', e)
            print('orgi path:', selected_image_path)
            print('Lime path:', limePath)
            print('shap path:', shapPath)
            return render_template("index.html", prediction=data["prediction"], selected_image=selected_image_path, lime_path=limePath, shap_path =shapPath)
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    # image_path = r"E:\Study\e17-4yp-Applying-deep-learning-on-histological-images-of-lymph-tissues\deployment\static\uploads\IHC.png"
    # print(generate_shap_mask(image_path))
    app.run(debug=True)
