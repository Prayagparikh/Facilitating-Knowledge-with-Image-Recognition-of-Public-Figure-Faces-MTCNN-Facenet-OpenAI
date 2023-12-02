from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import base64
from mtcnn import MTCNN
import joblib
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

predicted_labels_list = []

UPLOAD_FOLDER = './Final_Web_App/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
image_file = ''
app.config['image_file'] = image_file

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# Using Embeddings from the FACENET Face recognition model
from keras_facenet import FaceNet
facenet_model = FaceNet()

def get_embeddings(images):
    embeddings = []
    for image in images:
        image_resized = cv2.resize(image, (160,160))
        preprocessed_image = np.expand_dims(image_resized, axis=0)
        preprocessed_image = np.array(preprocessed_image)
        embedding = facenet_model.embeddings(preprocessed_image)[0]
        embeddings.append(embedding)
    return np.array(embeddings)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    # if file and allowed_file(file.filename):
    #     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    #     filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(filename)
    #     return 'File uploaded: {}'.format(file.filename)

    if file and allowed_file(file.filename):
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        new_filename = 'user_image.jpg'

        # Create the full path for the new file
        new_filepath = os.path.join(upload_folder, new_filename)

        # Remove all existing files in the upload folder
        for existing_file in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, existing_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        # Save the new file
        file.save(new_filepath)

        return 'File uploaded: {}'.format(new_filename)
    
    else:
        return 'Invalid file type. Please upload a .jpg, .jpeg, or .png file.'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}



@app.route('/detect_faces', methods=['POST'])
def face_detection():
    try:
        # input_image_path = "D:\\US admission docs\\UT Arlington\\Fall 2023\\Machine Learning\\notebook\\Final_Web_App\\uploaded_images\\user_image.jpg"
        input_image_path = os.path.join('Final_Web_App', 'uploaded_images', 'user_image.jpg')
        input_image = cv2.imread(input_image_path)

        if input_image is None:
            return jsonify({'error': 'Failed to load the input image.'}), 500
        
        print(input_image.shape)
        print(input_image.dtype)

        detector = MTCNN()

        # Detect faces in the input image
        faces = detector.detect_faces(input_image)

        faces_folder = './Final_Web_App/detected_faces'
        os.makedirs(faces_folder, exist_ok=True)

        # Clear existing images in the folder
        existing_files = os.listdir(faces_folder)
        for file in existing_files:
            file_path = os.path.join(faces_folder, file)
            os.remove(file_path)

        # List to store base64-encoded images
        detected_faces_base64 = []

        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            face_image = input_image[y:y+height, x:x+width]

            # Resize each detected face to (160, 160, 3)
            face_image = cv2.resize(face_image, (160, 160))

            # Convert the face image to base64
            _, encoded_face = cv2.imencode('.png', face_image)
            base64_data = base64.b64encode(encoded_face.tobytes()).decode('utf-8')
            detected_faces_base64.append(base64_data)

            # Save the detected face image
            face_filename = f'detected_face_{i}.png'
            face_path = os.path.join(faces_folder, face_filename)
            cv2.imwrite(face_path, face_image)  # Save as BGR

        # Return the base64-encoded face images along with the number of faces
        return jsonify({'detected_faces_base64': detected_faces_base64, 'num_faces': len(detected_faces_base64)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

    # image_path = "./uploaded_images/mypic.png"
    # return send_file(image_path, mimetype='image/png')

    # try:
    #     # Example image paths, replace with your logic to obtain actual image paths
    #     image_paths = [
    #         "D:\\US admission docs\\UT Arlington\\Fall 2023\\Machine Learning\\notebook\\Final_Web_App\\uploaded_images\\mypic.png",
    #         "D:\\US admission docs\\UT Arlington\\Fall 2023\\Machine Learning\\notebook\\Final_Web_App\\uploaded_images\\mypic.png"
    #     ]

    #     image_data_list = []

    #     for image_path in image_paths:
    #         with open(image_path, 'rb') as image_file:
    #             base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    #             image_data_list.append(base64_data)

    #     return jsonify({'image_data_list': image_data_list})

    # except Exception as e:
    #     print(f"Error: {e}")
    #     traceback.print_exc()  # Print the traceback for detailed error information
    #     return jsonify({'error': 'Internal Server Error'}), 500



@app.route('/recognize_faces', methods=['POST'])
def recognize_faces():
    global predicted_labels_list
    detected_faces_folder = './Final_Web_App/detected_faces'

    face_images = []
    # Iterate through files in the "detected_faces" folder
    for filename in os.listdir(detected_faces_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Construct the full path to the image
            image_path = os.path.join(detected_faces_folder, filename)

            # Read the image using OpenCV
            face_image = cv2.imread(image_path)

            # Append the image to the list
            print(face_image.shape, face_image.dtype)
            face_images.append(face_image)
    
    predicted_labels_list = face_images
    embeddings_detected_faces = get_embeddings(face_images)

    # Load the trained KNN classifier
    classifier = joblib.load('./train_classifier_knn.joblib')

    # Predict the labels for the detected faces
    predicted_labels = classifier.predict(embeddings_detected_faces)

    # Convert the numpy array to a regular Python list for JSON serialization
    predicted_labels_list = predicted_labels.tolist()

    # Return the predicted labels as a JSON response
    return jsonify({'predicted_labels': predicted_labels_list})


@app.route('/openai_text', methods=['POST'])
def openai_text():
    from queue import Queue
    from threading import Lock

    global predicted_labels_list

    if not predicted_labels_list:
        return jsonify({'error': 'No predicted labels available'})

    results = {}

    # Set your OpenAI GPT-3 API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.api_key = "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"

    # User's input prompt
    user_prompt = "Tell me interesting facts and contributions of {person_name} in maximum 5 lines that contain a maximum of 100 words."

    results_queue = Queue()
    lock = Lock()

    # Create a list to store messages for the conversation
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_completion(person_name):
        prompt_with_name = user_prompt.format(person_name=person_name)

        try:
            messages.append({"role": "user", "content": prompt_with_name})

            # Generate completion using GPT-3
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            # Extract and store the generated text for the current person
            result_text = completion['choices'][0]['message']['content']

            with lock:
                results_queue.put((person_name, result_text))

        except openai.error.OpenAIError as e:
            # Handle OpenAI errors
            print(f"Error for {person_name}: {e}")
            with lock:
                results_queue.put((person_name, f"Error: {str(e)}"))

        finally:
            # Remove the user message for the current person after the completion
            messages.pop()

    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor() as executor:
        # Submit tasks and gather future objects
        futures = [executor.submit(generate_completion, person) for person in predicted_labels_list]

    # Wait for all tasks to complete
    for future in as_completed(futures):
        # Handle exceptions if any
        try:
            future.result()
        except Exception as e:
            print(f"Exception: {e}")

    # Display results in the correct order
    while not results_queue.empty():
        person_name, result = results_queue.get()
        results[person_name] = result

    predicted_labels_list = []
    # Return results as JSON to the client
    print(results)
    return jsonify(results)

# if __name__ == '__main__':
#     app.run(port=int(os.environ.get('PORT', 8000)))

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)))
