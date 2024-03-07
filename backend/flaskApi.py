import io
import test_data
import faceRecognition
 
from flask import Flask,request,jsonify
import os
import shutil
import tempfile
import base64
from flask_cors import CORS
import pickle
import json
from moviepy.video.io.VideoFileClip import VideoFileClip

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        name = request.form.get('name')
        if not name:
            return jsonify({'status': 400, 'message': 'Name is required'})
        
        folder_path = os.path.join("Img_Collect", name)
        folder_path_cropped = os.path.join("Img_Collect", "cropped", name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for i in range(1, 30):
            image_key = f'image{i}'
            if image_key not in request.files:
                break  

            image_file = request.files[image_key]
            if image_file.filename == '':
                break

            image_data = image_file.read()

            with open(os.path.join(folder_path, f'{name}_{i}.jpg'), 'wb') as out_file:
                out_file.write(image_data)
        
        face_dict = faceRecognition.load()
        
        if name in face_dict and len(face_dict[name]) > 20:
            X, y = faceRecognition.getInput(face_dict)
            model = faceRecognition.ModelFit(X, y)
            return jsonify({'status': 200, 'message': 'Images saved successfully'})
        else:
            shutil.rmtree(folder_path)
            shutil.rmtree(folder_path_cropped)
            return jsonify({'status': 400, 'message': f'Only {len(face_dict[name])} images are in correct format. Please follow the instructions!'})
    
    except Exception as e:
        return jsonify({'status': 500, 'message': f'Internal server error: {str(e)}'})



@app.route('/check', methods=['POST'])
def saved():
    name = request.form.get('name')
    with open("backend/face_dict.json",'r') as f:
        face_dict = json.load(f)
    
    if name in face_dict: 
        return jsonify({"exists": True})
    else:
        return jsonify({"exists": False})
    


#prediction 
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return 'No video file provided'

    blob = request.files['video']
    byte_data = blob.read()
    encoded_data = base64.b64encode(byte_data)
    byte_data = base64.b64decode(encoded_data)

    with io.BytesIO(byte_data) as f:
        with open('test.mp4', 'wb') as out_file:
            shutil.copyfileobj(f, out_file)

    test_data.getImageFromVideo("test.mp4")
    face_dict =  test_data.load()
   
    url = "backend/test_dict.json"
    X_test = test_data.getTestInput(url)

    # Load model
    with open('backend/model.pkl', 'rb') as f:
        model = pickle.load(f)
   
    prediction = model.predict(X_test)
   
    # Load token
    with open('backend/token.json', 'r') as f:
        token = json.load(f)
    
    # Count predictions
    c = {}
    for j in token.values():
        c[j] = 0

    for p in prediction:
        if p in c:
            c[p] += 1
    
    max_value = max(c.values())
    name = [key for key, value in token.items() if c[value] == max_value]
    
    print(prediction)
    print(name[0])

    return jsonify({'message': name[0]})



if __name__ == "__main__":
           app.run()
