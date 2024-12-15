import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont

# Load model
model = load_model('animal_human_detector_model.keras')
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'human', 'sheep', 'spider', 'squirrel']

# Function to preprocess the image correctly
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 (matches model input size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (convert to float and scale)
    return img_array

# Initialize webcam
def capture_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not detected.")
        return

    os.makedirs('captured_images', exist_ok=True)  # Ensure folder exists

    print("Press 's' to save a frame, 'q' to quit, and 'p' to stop saving frames.")
    capturing = True  # Variable to control multiple captures

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        # Preprocess frame
        img_array = preprocess_image(frame)  # Correctly preprocess image

        # Predict
        predictions = model.predict(img_array)
        class_id = np.argmax(predictions[0])
        class_name = class_names[class_id]

        # Display results
        cv2.putText(frame, f'Detected: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam Feed', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and capturing:
            # Save the current frame
            timestamp = cv2.getTickCount()
            image_filename = f'captured_images/detected_{timestamp}.jpg'
            cv2.imwrite(image_filename, frame)
            print(f'Image saved: {image_filename}')

        elif key == ord('p'):
            # Toggle capturing
            capturing = not capturing
            state = 'enabled' if capturing else 'disabled'
            print(f'Saving frames {state}.')

        elif key == ord('q'):
            # Exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()

# Process images from a folder
def process_images_from_folder(folder_path):
    os.makedirs('captured_images', exist_ok=True)  # Ensure folder exists
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        if os.path.isfile(image_path):
            img = Image.open(image_path)
            img = img.resize((224, 224))  # Resize to match model input size
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

            # Predict
            predictions = model.predict(img_array)
            class_id = np.argmax(predictions[0])
            class_name = class_names[class_id]

            # Add text to the image
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()  # You can customize the font here
            text = f'Detected: {class_name}'
            draw.text((10, 10), text, font=font, fill='red')

            # Save the processed image
            processed_image_path = os.path.join('captured_images', f'processed_{image_file}')
            img.save(processed_image_path)
            print(f'{image_file}: {class_name} (Processed image saved as {processed_image_path})')

# Process a single image
def process_single_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Predict
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions[0])
    class_name = class_names[class_id]

    # Add text to the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # You can customize the font here
    text = f'Detected: {class_name}'
    draw.text((10, 10), text, font=font, fill='red')

    # Save the processed image
    processed_image_path = os.path.join('captured_images', f'processed_{os.path.basename(image_path)}')
    img.save(processed_image_path)
    print(f'{os.path.basename(image_path)}: {class_name} (Processed image saved as {processed_image_path})')

def main():
    print("Choose an option:")
    print("1. Capture from webcam")
    print("2. Process images from folder")
    print("3. Process a single image")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        capture_from_webcam()
    elif choice == '2':
        folder_path = input("Enter the path to the folder containing images: ")
        process_images_from_folder(folder_path)
    elif choice == '3':
        image_path = input("Enter the path to the image file: ")
        process_single_image(image_path)
    else:
        print("Invalid choice.")

if __name__ == '__main__':
    main()

