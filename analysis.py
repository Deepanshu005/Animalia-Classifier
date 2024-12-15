import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model(r'animalia_human_detector_model.keras')
model.summary()


# Define class names
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'human', 'sheep', 'spider', 'squirrel']

# Paths to test and train folders
train_folder = r'dataset\train'  # Update with your actual train data folder path
test_folder = r'dataset\test'  # Update with your actual test data folder path


# Function to process images and generate confusion matrix
def generate_confusion_matrix(data_folder, dataset_name):
    y_true = []
    y_pred = []

    for class_label, class_name in enumerate(class_names):
        class_folder = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_folder):
            print(f"Warning: {class_folder} is not a directory or does not exist.")
            continue
        
        for image_file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_file)
            
            # Load and preprocess the image
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

                # Ensure the input shape matches the model’s expected shape
                if img_array.size == 0:
                    print(f"Skipped {image_file}: Image array is empty.")
                    continue

                # Predict using the model
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])

                # Append true label and predicted label
                y_true.append(class_label)
                y_pred.append(predicted_class)
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Generate and display confusion matrix
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix for {dataset_name} Dataset")
        plt.tight_layout()
        plt.savefig(f"graphs/{dataset_name}_confusion_matrices.png")
        plt.show()
    else:
        print(f"No predictions were made for the {dataset_name} dataset. Check {data_folder} and image preprocessing.")

# Generate confusion matrix for the test dataset
generate_confusion_matrix(test_folder, "Test")


# Generate confusion matrix for the train dataset
generate_confusion_matrix(train_folder, "Train")



