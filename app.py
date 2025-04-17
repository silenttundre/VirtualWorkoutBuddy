import os
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
import io
import base64
import sys
sys.path.append('yolov5')

from pose_analyzer import PoseAnalyzer
from chatbot_feedback import ChatbotFeedback


# Initialize Flask application
app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize PoseAnalyzer
pose_analyzer = PoseAnalyzer()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(file_path):
    """Preprocess the image for analysis"""
    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return None
    
    # Resize if needed (keeping aspect ratio)
    max_dimension = 1024
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Apply basic enhancements
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge the CLAHE enhanced L channel with the original A and B channels
    limg = cv2.merge((cl, a, b))
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Save the preprocessed image
    preprocessed_path = file_path.replace('.', '_preprocessed.')
    cv2.imwrite(preprocessed_path, enhanced_image)
    
    return preprocessed_path

# Function to analyze pose in image
def analyze_pose(image_path, exercise_type='auto'):
    try:
        # Preprocess the image
        preprocessed_path = preprocess_image(image_path)
        if preprocessed_path is None:
            return {"error": "Could not read or preprocess image"}, None
        
        # Read the image
        image = cv2.imread(preprocessed_path)
        if image is None:
            return {"error": "Could not read preprocessed image"}, None
        
        # Detect pose landmarks
        landmarks, annotated_image, error = pose_analyzer.detect_pose(preprocessed_path)
        if error:
            return {"error": error}, None
        
        # Analyze exercise
        analysis_results, detected_exercise = pose_analyzer.analyze_exercise(landmarks, image)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Determine assessment
        status = "proper" if len(analysis_results[0]) == 0 else "improper"
        assessment = "Proper form! Great job!" if status == "proper" else "Improper form detected"
        
        # Clean up
        try:
            os.remove(preprocessed_path)
        except:
            pass
        
        return {
            "status": status,
            "assessment": assessment,
            "issues": analysis_results[0],
            "suggestions": analysis_results[1],
            "exercise_type": detected_exercise
        }, img_str
        
    except Exception as e:
        print(f"Error in analyze_pose: {str(e)}")
        return {"error": f"Analysis error: {str(e)}"}, None
    
# Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# Add these new routes to app.py
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# # Update the index route to redirect to home
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint hit")  # Debug logging
    if 'file' not in request.files:
        print("No file part in request")  # Debug logging
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    print(f"Received file: {file.filename}")  # Debug logging
    exercise_type = request.form.get('exercise_type', 'general')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        # Create a unique filename
        unique_id = str(uuid.uuid4())
        filename = f"{exercise_type}_{unique_id}_{int(time.time())}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze the pose
        analysis, annotated_image = analyze_pose(file_path, exercise_type)
        
        # Return the analysis and annotated image
        return jsonify({
            "analysis": analysis,
            "image": annotated_image,
            "exercise_type": exercise_type,
            "file_path": file_path
        })
    else:
        return jsonify({"error": "File type not allowed. Please upload an image (png, jpg, jpeg, gif)."})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# In app.py, modify the analyze_pose function:
def analyze_pose(image_path, exercise_type='auto'):
    # Preprocess the image
    preprocessed_path = preprocess_image(image_path)
    if preprocessed_path is None:
        return {"error": "Could not read image"}, None
    
    # Read the image for YOLO
    image = cv2.imread(preprocessed_path)
    
    # Analyze the pose using PoseAnalyzer
    landmarks, annotated_image, error = pose_analyzer.detect_pose(preprocessed_path)
    
    if error:
        return {"error": error}, None
    
    # Analyze exercise form (now with YOLO)
    if exercise_type == 'auto':
        analysis_results, detected_exercise = pose_analyzer.analyze_exercise(landmarks, image)
    else:
        analysis_results = pose_analyzer.analyze_exercise(landmarks, image, exercise_type)
        detected_exercise = exercise_type
    
    # Convert annotated image to base64 for display
    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Determine overall assessment
    if len(analysis_results[0]) == 0:
        assessment = "Proper form! Great job!"
        status = "proper"
    else:
        assessment = "Improper form detected. Please check the following issues:"
        status = "improper"
    
    # Clean up preprocessed image
    try:
        os.remove(preprocessed_path)
    except:
        pass
    
    return {
        "status": status,
        "assessment": assessment,
        "issues": analysis_results[0],
        "suggestions": analysis_results[1],
        "exercise_type": detected_exercise,
        "detected_exercise": detected_exercise
    }, img_str

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
