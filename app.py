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
import requests
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
# In app.py after initializing pose_analyzer
pose_analyzer.debug_references()
chatbot_feedback = ChatbotFeedback()

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL

@app.route('/chat', methods=['POST'])
def chat_with_rag():
    """Handle chatbot messages with RAG from Ollama"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Prepare the prompt with workout context
        workout_context = """
        You are Virtual Workout Buddy, an AI fitness assistant. You help users with:
        - Exercise form analysis and corrections
        - Workout routines and programming
        - Fitness advice and guidance
        - Nutrition tips for athletes
        - Injury prevention and recovery
        - Motivation and progress tracking
        
        You have access to workout analysis data and can help users understand their form feedback.
        Be encouraging, professional, and focus on safe exercise practices.
        """
        
        full_prompt = f"{workout_context}\n\nUser: {user_message}\nAssistant:"
        
        # Call Ollama API
        ollama_payload = {
            "model": "llama2",  # or whatever model you have installed
            "prompt": full_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                ollama_response = response.json()
                bot_response = ollama_response.get('response', 'I apologize, but I cannot process your request right now.')
                
                return jsonify({"response": bot_response})
            else:
                return jsonify({"response": "I'm currently unavailable. Please try the workout analysis feature or check back later."})
                
        except requests.exceptions.RequestException:
            # Fallback responses if Ollama is not available
            return handle_fallback_chat(user_message)
            
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"response": "I'm experiencing technical difficulties. Please try again later."})

def handle_fallback_chat(message):
    """Provide fallback responses when Ollama is not available"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return jsonify({"response": "Hello! I'm Virtual Workout Buddy. While my advanced features are temporarily unavailable, I can still help you with basic workout questions and form analysis uploads."})
    
    elif any(word in message_lower for word in ['upload', 'image', 'photo']):
        return jsonify({"response": "You can upload workout images for analysis by clicking the 'Try It Now' button on the main page. Make sure to select the correct exercise type from the dropdown menu!"})
    
    elif any(word in message_lower for word in ['exercise', 'workout']):
        return jsonify({"response": "I support analysis for squats, push-ups, planks, lunges, deadlifts, and general posture. Upload an image of your exercise form for detailed feedback!"})
    
    elif any(word in message_lower for word in ['form', 'technique']):
        return jsonify({"response": "Proper form is crucial for effective workouts and injury prevention. Upload an image of your exercise, and I'll analyze your alignment, posture, and technique."})
    
    else:
        return jsonify({"response": "I specialize in workout form analysis and fitness guidance. You can upload exercise images for instant feedback, or ask me about specific exercises and proper techniques."})

# Health check for Ollama
@app.route('/health/ollama')
def check_ollama_health():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return jsonify({"status": "connected", "models": response.json().get('models', [])})
    except:
        return jsonify({"status": "disconnected"}), 503

@app.route('/debug_references')
def debug_references():
    """Debug endpoint to see what references are loaded"""
    try:
        debug_info = {
            'reference_positions': {},
            'total_references': 0
        }
        
        for exercise, qualities in pose_analyzer.reference_positions.items():
            debug_info['reference_positions'][exercise] = {}
            for quality, references in qualities.items():
                debug_info['reference_positions'][exercise][quality] = {
                    'count': len(references),
                    'files': [ref['file'] for ref in references],
                    'files_exist': [os.path.exists(ref['file']) for ref in references]
                }
                debug_info['total_references'] += len(references)
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/check_file_exists', methods=['POST'])
def check_file_exists():
    """Check if specific files exist"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({"error": "No file path provided"})
        
        exists = os.path.exists(file_path)
        return jsonify({
            "file_path": file_path,
            "exists": exists,
            "size": os.path.getsize(file_path) if exists else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)})

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
    """Handle training image uploads - simplified version without complex renaming"""
    try:
        print("=== UPLOAD TRAINING STARTED ===")
        
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"})
        
        files = request.files.getlist('file')
        exercise_type = request.form.get('exercise_type')
        form_quality = request.form.get('form_quality')
        
        print(f"Received {len(files)} files for {exercise_type} - {form_quality}")
        
        if not exercise_type or not form_quality:
            return jsonify({"success": False, "error": "Exercise type and form quality are required"})
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({"success": False, "error": "No selected files"})
        
        successful_uploads = []
        errors = []
        
        for i, file in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            if file and file.filename and allowed_file(file.filename):
                try:
                    # Create a simple temporary filename
                    temp_filename = f"temp_{int(time.time())}_{i}_{file.filename}"
                    temp_path = os.path.join(app.config['TRAINING_FOLDER'], temp_filename)
                    
                    print(f"Saving temporary file to: {temp_path}")
                    
                    # Save the file
                    file.save(temp_path)
                    
                    # Verify the file was saved correctly
                    if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                        errors.append(f"{file.filename}: File failed to save")
                        continue
                    
                    file_size = os.path.getsize(temp_path)
                    print(f"File saved successfully: {file_size} bytes")
                    
                    # Add to reference images using the pose analyzer
                    try:
                        reference_path = pose_analyzer.add_reference_image(
                            temp_path, exercise_type, form_quality
                        )
                        
                        if reference_path and os.path.exists(reference_path):
                            successful_uploads.append({
                                "filename": file.filename,
                                "reference_path": reference_path
                            })
                            print(f"Successfully added reference: {reference_path}")
                        else:
                            errors.append(f"{file.filename}: Reference image was not created properly")
                        
                    except Exception as ref_error:
                        errors.append(f"{file.filename}: {str(ref_error)}")
                        print(f"Reference creation error: {ref_error}")
                    
                    # Clean up temporary file regardless of success
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            print(f"Cleaned up temporary file: {temp_path}")
                    
                except Exception as file_error:
                    error_msg = f"{file.filename}: {str(file_error)}"
                    errors.append(error_msg)
                    print(f"File processing error: {error_msg}")
                    
                    # Clean up temporary file on error
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                error_msg = f"{file.filename if file and file.filename else 'Unknown file'}: Invalid file type or empty filename"
                errors.append(error_msg)
        
        print(f"Upload completed: {len(successful_uploads)} successful, {len(errors)} errors")
        
        # Return detailed response
        if successful_uploads:
            response_data = {
                "success": True,
                "message": f"Successfully added {len(successful_uploads)} reference image(s)",
                "uploaded_count": len(successful_uploads)
            }
            
            if errors:
                response_data["errors"] = errors
                response_data["has_errors"] = True
                
            return jsonify(response_data)
        else:
            return jsonify({
                "success": False,
                "error": "No files were uploaded successfully",
                "details": errors
            })
            
    except Exception as e:
        error_msg = f"Upload processing error: {str(e)}"
        print(error_msg)
        return jsonify({"success": False, "error": error_msg})

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