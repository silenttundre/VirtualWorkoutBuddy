# app.py
import os
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
import cv2
import numpy as np
import base64
from PIL import Image
import io
import json
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our pose analyzer and chatbot
from pose_analyzer import EnhancedPoseAnalyzer
from chatbot_feedback import ChatbotFeedback

# Initialize Flask application
app = Flask(__name__, static_folder='static')
app.secret_key = 'your-secret-key-here'  # Change this in production

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRAINING_FOLDER'] = 'training_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)

# Initialize PoseAnalyzer and ChatbotFeedback
pose_analyzer = EnhancedPoseAnalyzer()
chatbot_feedback = ChatbotFeedback()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(file_path):
    """Preprocess the image for analysis"""
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
    
    return file_path  # Return original path for now

def determine_final_status(issues, reference_similarity=0.0):
    """Determine final status with clear color coding"""
    if len(issues) == 0:
        return "proper", "Excellent form! Perfect technique! 🎉"
    
    # More generous classification
    if reference_similarity > 0.75 and len(issues) <= 1:
        return "proper", "Excellent form matching proper examples! ✅"
    elif reference_similarity > 0.65 and len(issues) <= 1:
        return "good", "Good form with minor adjustments needed! 👍"
    elif reference_similarity > 0.6 and len(issues) <= 2:
        return "good", "Good form showing proper technique patterns! 👌"
    elif len(issues) <= 2:
        return "needs_work", "Form needs some improvement 💪"
    elif len(issues) <= 3:
        return "needs_work", "Several areas need attention 📝"
    else:
        return "improper", "Form needs significant improvement 🔴"

def analyze_pose(image_path, exercise_type='auto'):
    """Analyze pose in image using enhanced analyzer"""
    try:
        # Preprocess the image
        preprocessed_path = preprocess_image(image_path)
        if preprocessed_path is None:
            return {"error": "Could not read or preprocess image"}, None
        
        # Read the image for analysis
        image = cv2.imread(preprocessed_path)
        if image is None:
            return {"error": "Could not read image"}, None
        
        # Detect pose landmarks
        landmarks, annotated_image, error = pose_analyzer.detect_pose(preprocessed_path)
        if error:
            return {"error": error}, None
        
        # ALWAYS use the user-selected exercise type
        detected_exercise = exercise_type if exercise_type not in ['general', 'auto'] else 'general'
        
        print(f"User selected exercise: {exercise_type}")
        print(f"Using exercise type for analysis: {detected_exercise}")
        
        # Call the enhanced method with user selection
        analysis_results, final_exercise = pose_analyzer.enhanced_analyze_exercise(
            landmarks, preprocessed_path, detected_exercise
        )
        
        # Apply visual feedback to the annotated image - PASS THE STATUS
        if landmarks and analysis_results:
            # Determine status first
            reference_match, similarity_score = pose_analyzer.compare_with_references_enhanced(landmarks, detected_exercise)
            
            # Use the same logic as below to determine status
            if reference_match == "proper" and similarity_score > 0.6:
                status = "good"
            elif reference_match == "proper" and similarity_score > 0.5:
                status = "needs_work"
            else:
                status = determine_final_status(analysis_results[0], similarity_score)[0]  # NOW THIS WILL WORK
            
            annotated_image = pose_analyzer.draw_posture_feedback(
                annotated_image, landmarks, analysis_results[0], status
            )
            
            # Store for potential use in the drawing
            pose_analyzer.last_similarity_score = similarity_score

        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Get reference similarity for better assessment
        reference_match, similarity_score = pose_analyzer.compare_with_references_enhanced(landmarks, detected_exercise)
        print(f"Reference match: {reference_match}, similarity score: {similarity_score:.2f}")

        # Determine final status - BE MORE GENEROUS
        if reference_match == "proper" and similarity_score > 0.6:
            status, assessment = "good", "Good form matching proper examples!"
        elif reference_match == "proper" and similarity_score > 0.5:
            status, assessment = "needs_work", "Form shows similarity to proper examples with room for improvement"
        else:
            status, assessment = determine_final_status(analysis_results[0], similarity_score)  # NOW THIS WILL WORK
        
        # Generate chatbot feedback
        feedback_data = {
            "status": status,
            "issues": analysis_results[0],
            "suggestions": analysis_results[1],
            "exercise_type": final_exercise,
            "reference_similarity": float(similarity_score)  # Convert to native float
        }
        
        print(f"DEBUG: Feedback data sent to chatbot: {feedback_data}")
        
        feedback_messages = chatbot_feedback.generate_form_feedback(feedback_data)
        
        print(f"DEBUG: Chatbot generated messages: {feedback_messages}")
        
        # Create the response with all native Python types
        response_data = {
            "status": status,
            "assessment": assessment,
            "issues": analysis_results[0],
            "suggestions": analysis_results[1],
            "exercise_type": final_exercise,
            "reference_similarity": float(similarity_score),  # Convert to native float
            "feedback": feedback_messages
        }
        
        print(f"DEBUG: Final response data: {response_data}")
        
        return response_data, img_str
        
    except Exception as e:
        print(f"Error in analyze_pose: {str(e)}")
        return {"error": f"Analysis error: {str(e)}"}, None

# Routes
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/training')
def training():
    """Training page for adding reference images"""
    stats = pose_analyzer.get_training_statistics()
    return render_template('training.html', stats=stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle regular workout analysis uploads"""
    print("Upload endpoint hit")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
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
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        if analysis.get("error"):
            return jsonify({"error": analysis["error"]})
        
        return jsonify({
            "analysis": analysis,
            "image": annotated_image
        })
    
    return jsonify({"error": "File type not allowed"})

@app.route('/upload_training', methods=['POST'])
def upload_training_image():
    """Handle training image uploads - support multiple files"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        files = request.files.getlist('file')  # Get list of files
        exercise_type = request.form.get('exercise_type')
        form_quality = request.form.get('form_quality')  # 'proper' or 'improper'
        
        if not exercise_type or not form_quality:
            return jsonify({"error": "Exercise type and form quality are required"})
        
        if not files or files[0].filename == '':
            return jsonify({"error": "No selected files"})
        
        successful_uploads = []
        errors = []
        processed_files = set()  # Track processed files to avoid duplicates
        
        for file in files:
            if file and allowed_file(file.filename):
                # Skip if we've already processed this file (duplicate check)
                # Use a more reliable identifier
                file_identifier = f"{file.filename}_{file.content_length or 0}"
                if file_identifier in processed_files:
                    errors.append(f"{file.filename}: Duplicate file skipped")
                    continue
                
                processed_files.add(file_identifier)
                
                # Save the uploaded file temporarily
                filename = secure_filename(file.filename)
                unique_id = str(uuid.uuid4())
                temp_filename = f"temp_{unique_id}_{filename}"
                temp_path = os.path.join(app.config['TRAINING_FOLDER'], temp_filename)
                
                try:
                    file.save(temp_path)
                    
                    # Check if file was saved successfully
                    if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                        errors.append(f"{file.filename}: File upload failed or empty file")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        continue
                    
                    # Check if this exact file already exists in references to avoid true duplicates
                    file_hash = None
                    try:
                        import hashlib
                        with open(temp_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                    except:
                        pass
                    
                    # Add to reference images
                    reference_path = pose_analyzer.add_reference_image(
                        temp_path, exercise_type, form_quality
                    )
                    
                    # Verify the reference was actually added
                    if reference_path and os.path.exists(reference_path):
                        successful_uploads.append({
                            "filename": file.filename,
                            "reference_path": reference_path
                        })
                    else:
                        errors.append(f"{file.filename}: Failed to add as reference image - reference path not created")
                    
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                except Exception as e:
                    # Clean up temporary file on error
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    errors.append(f"{file.filename}: {str(e)}")
                    print(f"Error processing {file.filename}: {str(e)}")
            else:
                errors.append(f"{file.filename}: File type not allowed or invalid file")
        
        # Return detailed response
        if successful_uploads:
            response_data = {
                "success": True,
                "message": f"Successfully added {len(successful_uploads)} reference image(s)",
                "uploads": successful_uploads,
                "errors": errors
            }
            # If there are any errors, include them even if some uploads were successful
            if errors:
                response_data["has_errors"] = True
                response_data["message"] += f" ({len(errors)} files had errors)"
            return jsonify(response_data)
        else:
            return jsonify({
                "success": False,
                "error": "No files were uploaded successfully", 
                "details": errors
            })
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"success": False, "error": f"Upload error: {str(e)}"})

@app.route('/debug_upload', methods=['POST'])
def debug_upload():
    """Debug endpoint to see what files are being received"""
    try:
        debug_info = {
            'files_received': [],
            'form_data': dict(request.form)
        }
        
        if 'file' in request.files:
            files = request.files.getlist('file')
            for i, file in enumerate(files):
                if file and file.filename:
                    debug_info['files_received'].append({
                        'index': i,
                        'filename': file.filename,
                        'content_length': file.content_length,
                        'content_type': file.content_type,
                        'headers': dict(file.headers)
                    })
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/training_stats')
def training_stats():
    """Get training statistics"""
    try:
        stats = pose_analyzer.get_training_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/delete_reference', methods=['POST'])
def delete_reference():
    """Delete a reference image"""
    try:
        data = request.get_json()
        exercise_type = data.get('exercise_type')
        form_quality = data.get('form_quality')
        file_path = data.get('file_path')
        
        if not all([exercise_type, form_quality, file_path]):
            return jsonify({"error": "Missing required parameters"})
        
        # Remove from reference data
        references = pose_analyzer.reference_positions.get(exercise_type, {}).get(form_quality, [])
        pose_analyzer.reference_positions[exercise_type][form_quality] = [
            ref for ref in references if ref['file'] != file_path
        ]
        
        # Delete physical file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass  # File might already be deleted
        
        # Save updated metadata
        pose_analyzer.save_reference_metadata()
        
        return jsonify({"success": True, "message": "Reference image deleted"})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/reference_images/<exercise>/<quality>')
def get_reference_images(exercise, quality):
    """Get list of reference images for display"""
    try:
        references = pose_analyzer.reference_positions.get(exercise, {}).get(quality, [])
        
        # Convert images to base64 for display
        image_data = []
        for ref in references:
            try:
                if os.path.exists(ref['file']):
                    with open(ref['file'], "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                        image_data.append({
                            'image': f"data:image/jpeg;base64,{encoded_image}",
                            'file_path': ref['file'],
                            'timestamp': ref['timestamp'],
                            'exercise': exercise,  # Add exercise explicitly
                            'quality': quality     # Add quality explicitly
                        })
            except Exception as e:
                print(f"Error loading image {ref['file']}: {e}")
                continue
        
        return jsonify(image_data)
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)