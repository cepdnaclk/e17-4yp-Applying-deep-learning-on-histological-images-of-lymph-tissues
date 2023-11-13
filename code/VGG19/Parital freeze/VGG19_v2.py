import math
from discordwebhook import Discord

def DiscordNotification(Msg):
    webHookUrl = "https://discord.com/api/webhooks/1132597585824202813/8XDNjpwwOIsistL4nThyY7NjVo67UVHckbtOAAdGAf96_TZ7dTS3tOpDmle646rF_ZDX"
    discord = Discord(url=webHookUrl)
    discord.post(content=Msg)
try:
    import os
    directory = "prediction sample"
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass  # Directory already exists
    import math
    from discordwebhook import Discord
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import learning_curve
    from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess_input
    
    import seaborn as sns
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Define constants
    input_shape = (224, 224, 3)
    num_classes = 2
    epochs = 50  # Number of epochs for training

    # Load pre-trained VGG19 model (excluding top classification layer)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    train_data = "/home/e17256/fyp/dataset/Breast Histopathology Images new/train"
    vali_data = "/home/e17256/fyp/dataset/Breast Histopathology Images new/valid"
    test_data = "/home/e17256/fyp/dataset/Breast Histopathology Images new/test"
    
    # train_data = "/home/e17256/fyp/dataset/Temp/train"
    # vali_data = "/home/e17256/fyp/dataset/Temp/validate"
    # test_data = "/home/e17256/fyp/dataset/Temp/test"
    # Create a new model on top of the pre-trained base model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # To freeze part of the model
    for layer in base_model.layers[:10]:
        layer.trainable = False

    for layer in base_model.layers[10:]:
        layer.trainable = True

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess data using ImageDataGenerator for training
    train_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data,  # Path to the directory containing '0' and '1' folders
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Load and preprocess data using ImageDataGenerator for validation
    validation_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess_input)

    validation_generator = validation_datagen.flow_from_directory(
        vali_data,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Train the model with validation and get the training history
    history = model.fit(train_generator, epochs=epochs, steps_per_epoch=len(train_generator),
                        validation_data=validation_generator, validation_steps=len(validation_generator))

    # Save the trained model
    model.save('trained_model.h5')
    test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        test_data,  # Path to the directory containing '0' and '1' folders
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    # Plot and save Epoch vs. Training Accuracy and Validation Accuracy graphs
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs. Accuracy')
    plt.legend()
    plt.savefig('epoch_accuracy.png')
    plt.clf()

    # Plot and save Epoch vs. Training Loss and Validation Loss graphs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs. Loss')
    plt.legend()
    plt.savefig('epoch_loss.png')
    plt.clf()

    y_true = test_generator.classes
    # Compute ROC curve and AUC
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot and save ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.clf()

    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
    avg_precision = average_precision_score(y_true, y_pred_probs[:, 1])

    # Plot and save precision-recall curve
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: Avg. Precision={0:0.2f}'.format(avg_precision))
    plt.savefig('precision_recall_curve.png')
    plt.clf()

    # Access the learning rate schedule (if you're using a custom scheduler)
    
    learning_rates = [0.0001]*epochs

    # Plot and save learning rate schedule
    plt.plot(range(epochs), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig('learning_rate_schedule.png')
    plt.clf()

    # Plot comparison of validation metrics
    plt.plot(history.history['val_accuracy'], label='Model 1 Accuracy')
    plt.plot(history.history['val_accuracy'], label='Model 2 Accuracy')
    # Add more lines for other metrics or models
    plt.xlabel('Epoch')
    plt.ylabel('Validation Metric')
    plt.title('Validation Metrics Comparison')
    plt.legend()
    plt.savefig('validation_metrics_comparison.png')
    plt.clf()

    # # Plot learning curve
    # train_sizes, train_scores, val_scores = learning_curve(
    # model, train_generator, y_true, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

    # # Calculate mean and standard deviation of train scores and validation scores
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # val_scores_mean = np.mean(val_scores, axis=1)
    # val_scores_std = np.std(val_scores, axis=1)
    # plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
    # plt.plot(train_sizes, val_scores_mean, label='Validation Accuracy')
    # plt.xlabel('Training Dataset Size')
    # plt.ylabel('Accuracy')
    # plt.title('Learning Curve')
    # plt.legend()
    # plt.savefig('learning_curve.png')
    # plt.clf()

    # Predict and visualize a few test samples
    num_samples = 20  # Number of samples to visualize
    for i in range(num_samples):
        batch = next(test_generator)  # Get the next batch from the generator
        images, labels = batch  # Unpack the batch into images and labels
        true_label = np.argmax(labels[0])  # Get the true label of the first image in the batch
        predicted_probs = model.predict(images)  # Predict probabilities for the batch
        predicted_label = np.argmax(predicted_probs[0])  # Get the predicted label of the first image

        plt.figure()
        plt.imshow(images[0])  # Display the image
        plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
        plt.savefig(f'prediction sample/prediction_sample_{i}.png')
        plt.clf()
        
    confusion = confusion_matrix(y_true, y_pred)
    
    # Create a heatmap for the confusion matrix
    class_names = ["0","1"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.clf()
    
    DiscordNotification("FYP(vgg19Test) Graphs and Evaluations Completed")

except Exception as e:
    print("Error: ", e)
    DiscordNotification(f"FYP(vgg19Test) Error: {e}")
