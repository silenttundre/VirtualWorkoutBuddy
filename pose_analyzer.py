# pose_analyzer.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import json
from datetime import datetime
from pathlib import Path
import logging
from scipy.spatial import distance
import math

class EnhancedPoseAnalyzer:
    def __init__(self):
        # Setup logging FIRST
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Exercise classes
        self.exercise_classes = ['squat', 'pushup', 'plank', 'lunge', 'deadlift']
        
        # YOLO pose keypoint mapping (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Body part ratios and angle thresholds
        self.posture_thresholds = {
            'squat': {
                'back_arch_threshold': 25,  # Curvature threshold for arched back
                'knee_forward_threshold': 0.8,  # Ratio of knee to toe
                'hip_drop_threshold': 0.3  # Hip depth ratio
            },
            'pushup': {
                'body_alignment_threshold': 10,  # Degrees from horizontal
                'elbow_angle_range': (90, 120)  # Proper elbow angle range
            }
        }
        
        # Setup training data directories
        self.setup_training_directories()
        
        # Initialize YOLO models AFTER logger is set up
        self.yolo_pose_model = self.load_yolo_pose_model()
        self.yolo_detection_model = self.load_yolo_detection_model()
        
        # Load reference positions
        self.reference_positions = self.load_reference_positions()
        
        self.logger.info("EnhancedPoseAnalyzer initialized successfully")
    
    def setup_training_directories(self):
        """Create directories for training data"""
        self.training_root = Path("training_data")
        self.reference_root = Path("reference_positions")
        
        # Create main directories
        for directory in [self.training_root, self.reference_root]:
            directory.mkdir(exist_ok=True)
            
            # Create subdirectories for each exercise
            for exercise in self.exercise_classes:
                (directory / exercise).mkdir(exist_ok=True)
                (directory / exercise / "proper").mkdir(exist_ok=True)
                (directory / exercise / "improper").mkdir(exist_ok=True)
    
    def load_yolo_pose_model(self):
        """Load YOLOv11 pose estimation model"""
        try:
            # Use YOLOv11 pose model (downloads automatically on first use)
            model = YOLO('yolo11n-pose.pt')  # nano pose model
            self.logger.info("YOLOv11 pose model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading YOLOv11 pose model: {e}")
            # Fallback to regular detection model
            try:
                model = YOLO('yolo11n.pt')
                self.logger.info("Fallback to YOLOv11 detection model")
                return model
            except Exception as e2:
                self.logger.error(f"Error loading fallback model: {e2}")
                return None
    
    def load_yolo_detection_model(self):
        """Load YOLOv11 object detection model"""
        try:
            model = YOLO('yolo11n.pt')  # nano detection model
            self.logger.info("YOLOv11 detection model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading YOLOv11 detection model: {e}")
            return None
    
    def load_reference_positions(self):
        """Load reference position data for comparison"""
        reference_data = {}
        
        for exercise in self.exercise_classes:
            reference_data[exercise] = {
                "proper": [],
                "improper": []
            }
            
            # Load proper reference images
            proper_dir = self.reference_root / exercise / "proper"
            if proper_dir.exists():
                for img_file in proper_dir.glob("*.jpg"):
                    try:
                        keypoints = self.extract_keypoints_from_image(str(img_file))
                        if keypoints:
                            reference_data[exercise]["proper"].append({
                                "keypoints": keypoints,
                                "file": str(img_file),
                                "timestamp": datetime.now().isoformat()
                            })
                    except Exception as e:
                        self.logger.error(f"Error loading reference image {img_file}: {e}")
        
        return reference_data
    
    def extract_keypoints_from_image(self, image_path):
        """Extract pose keypoints from an image using YOLOv11"""
        if self.yolo_pose_model is None:
            return None
        
        try:
            # Run YOLO pose detection
            results = self.yolo_pose_model(image_path, verbose=False)
            
            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Get the first person's keypoints
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    if len(keypoints_data) > 0:
                        keypoints = keypoints_data[0]  # Shape: [17, 3] (x, y, confidence)
                        
                        # Convert to list format for JSON serialization
                        keypoints_list = []
                        for i, (x, y, conf) in enumerate(keypoints):
                            keypoints_list.append({
                                "name": self.keypoint_names[i],
                                "x": float(x),
                                "y": float(y),
                                "confidence": float(conf)
                            })
                        
                        return keypoints_list
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting keypoints from {image_path}: {e}")
            return None
    
    def detect_pose(self, image_path):
        """Detect human pose using YOLOv11"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, None, "Could not read image"
            
            # Extract keypoints using YOLO
            keypoints = self.extract_keypoints_from_image(image_path)
            
            # Create annotated image
            annotated_image = image.copy()
            if keypoints:
                annotated_image = self.draw_keypoints(annotated_image, keypoints)
            
            return keypoints, annotated_image, None
        except Exception as e:
            return None, None, str(e)
    
    def draw_keypoints(self, image, keypoints):
        """Draw keypoints and skeleton on image"""
        if not keypoints:
            return image
        
        # Define skeleton connections (COCO format)
        skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
        ]
        
        # Convert keypoints to format for drawing
        points = []
        for kp in keypoints:
            if kp['confidence'] > 0.5:  # Only draw confident keypoints
                x, y = int(kp['x']), int(kp['y'])
                points.append((x, y))
                # Draw keypoint
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            else:
                points.append(None)
        
        # Draw skeleton
        for connection in skeleton:
            if (connection[0] < len(points) and connection[1] < len(points) and 
                points[connection[0]] is not None and points[connection[1]] is not None):
                cv2.line(image, points[connection[0]], points[connection[1]], (255, 0, 0), 2)
        
        return image
    
    def add_reference_image(self, image_path, exercise_type, form_quality):
        """Add a new reference image for training with better error handling"""
        if exercise_type not in self.exercise_classes:
            raise ValueError(f"Invalid exercise type: {exercise_type}")
        
        if form_quality not in ["proper", "improper"]:
            raise ValueError(f"Invalid form quality: {form_quality}")
        
        try:
            # Verify the source image exists and is readable
            if not os.path.exists(image_path):
                raise ValueError(f"Source image does not exist: {image_path}")
            
            # Extract keypoints
            keypoints = self.extract_keypoints_from_image(image_path)
            if not keypoints:
                raise ValueError("Could not extract pose keypoints from image - no human pose detected")
            
            # Count how many confident keypoints we have
            confident_keypoints = [kp for kp in keypoints if kp['confidence'] > 0.5]
            if len(confident_keypoints) < 8:  # Need at least 8 confident keypoints
                raise ValueError(f"Insufficient pose detection (only {len(confident_keypoints)} confident keypoints)")
            
            # Create destination path with unique name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            original_name = os.path.basename(image_path)
            safe_name = secure_filename(original_name)
            filename = f"ref_{timestamp}_{safe_name}"
            dest_path = self.reference_root / exercise_type / form_quality / filename
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy image to reference directory
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read source image with OpenCV")
            
            success = cv2.imwrite(str(dest_path), image)
            if not success:
                raise ValueError("Failed to save reference image")
            
            # Verify the image was saved
            if not os.path.exists(dest_path):
                raise ValueError("Reference image was not created")
            
            # Add to reference data
            if exercise_type not in self.reference_positions:
                self.reference_positions[exercise_type] = {"proper": [], "improper": []}
            
            reference_data = {
                "keypoints": keypoints,
                "file": str(dest_path),
                "timestamp": datetime.now().isoformat(),
                "keypoint_count": len(confident_keypoints)
            }
            
            self.reference_positions[exercise_type][form_quality].append(reference_data)
            
            # Save reference data
            self.save_reference_metadata()
            
            self.logger.info(f"Added reference image: {dest_path} with {len(confident_keypoints)} keypoints")
            return str(dest_path)
            
        except Exception as e:
            self.logger.error(f"Error adding reference image {image_path}: {e}")
            # Clean up any partially created files
            if 'dest_path' in locals() and os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except:
                    pass
            raise
    def add_reference_image(self, image_path, exercise_type, form_quality):
        """Add a new reference image for training with better error handling"""
        if exercise_type not in self.exercise_classes:
            raise ValueError(f"Invalid exercise type: {exercise_type}")
        
        if form_quality not in ["proper", "improper"]:
            raise ValueError(f"Invalid form quality: {form_quality}")
        
        try:
            # Verify the source image exists and is readable
            if not os.path.exists(image_path):
                raise ValueError(f"Source image does not exist: {image_path}")
            
            # Extract keypoints
            keypoints = self.extract_keypoints_from_image(image_path)
            if not keypoints:
                raise ValueError("Could not extract pose keypoints from image - no human pose detected")
            
            # Count how many confident keypoints we have
            confident_keypoints = [kp for kp in keypoints if kp['confidence'] > 0.5]
            if len(confident_keypoints) < 8:  # Need at least 8 confident keypoints
                raise ValueError(f"Insufficient pose detection (only {len(confident_keypoints)} confident keypoints)")
            
            # Create destination path - KEEP THE ORIGINAL FILENAME
            original_name = os.path.basename(image_path)
            dest_path = self.reference_root / exercise_type / form_quality / original_name
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If file already exists, add a simple counter to avoid overwriting
            counter = 1
            base_name = os.path.splitext(original_name)[0]
            extension = os.path.splitext(original_name)[1]
            
            while dest_path.exists():
                new_name = f"{base_name}_{counter}{extension}"
                dest_path = self.reference_root / exercise_type / form_quality / new_name
                counter += 1
            
            # Copy image to reference directory
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read source image with OpenCV")
            
            success = cv2.imwrite(str(dest_path), image)
            if not success:
                raise ValueError("Failed to save reference image")
            
            # Verify the image was saved
            if not os.path.exists(dest_path):
                raise ValueError("Reference image was not created")
            
            # Add to reference data
            if exercise_type not in self.reference_positions:
                self.reference_positions[exercise_type] = {"proper": [], "improper": []}
            
            reference_data = {
                "keypoints": keypoints,
                "file": str(dest_path),
                "timestamp": datetime.now().isoformat(),
                "keypoint_count": len(confident_keypoints)
            }
            
            self.reference_positions[exercise_type][form_quality].append(reference_data)
            
            # Save reference data
            self.save_reference_metadata()
            
            self.logger.info(f"Added reference image: {dest_path} with {len(confident_keypoints)} keypoints")
            return str(dest_path)
            
        except Exception as e:
            self.logger.error(f"Error adding reference image {image_path}: {e}")
            # Clean up any partially created files
            if 'dest_path' in locals() and os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except:
                    pass
            raise

    def save_reference_metadata(self):
        """Save reference position metadata to JSON file"""
        metadata_file = self.reference_root / "reference_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(self.reference_positions, f, indent=2, default=str)
    
    def detect_exercise_yolov11(self, image_path):
        """Detect exercise type using pose-based heuristics"""
        try:
            keypoints = self.extract_keypoints_from_image(image_path)
            if not keypoints:
                return 'general'
            
            # Convert keypoints to dictionary for easier access
            kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
            
            # Simple heuristics based on pose
            if all(key in kp_dict for key in ['left_shoulder', 'left_hip', 'left_knee', 'left_wrist']):
                left_shoulder = kp_dict['left_shoulder']
                left_hip = kp_dict['left_hip']
                left_knee = kp_dict['left_knee']
                left_wrist = kp_dict['left_wrist']
                
                # Check if person is in plank position (horizontal body, arms extended)
                if (abs(left_shoulder['y'] - left_hip['y']) < 100 and 
                    left_wrist['y'] < left_shoulder['y']):
                    return 'plank'
                
                # Check if person is squatting (knees bent significantly)
                knee_bend = left_hip['y'] - left_knee['y']
                if knee_bend > 50:
                    return 'squat'
                
                # Check if person is doing push-ups (similar to plank but different arm position)
                if (left_wrist['y'] < left_shoulder['y'] and 
                    abs(left_shoulder['y'] - left_hip['y']) < 150):
                    return 'pushup'
            
            return 'general'
            
        except Exception as e:
            self.logger.error(f"Exercise detection error: {e}")
            return 'general'
    
    def compare_with_references(self, keypoints, exercise_type):
        """Enhanced reference comparison with better scoring"""
        if exercise_type not in self.reference_positions:
            return None, 0.0
        
        references = self.reference_positions[exercise_type]
        if not references["proper"] and not references["improper"]:
            return None, 0.0
        
        try:
            # Calculate similarity with proper references
            proper_similarities = []
            for ref in references["proper"]:
                similarity = self.calculate_enhanced_keypoint_similarity(keypoints, ref["keypoints"])
                proper_similarities.append(similarity)
            
            # Calculate similarity with improper references
            improper_similarities = []
            for ref in references["improper"]:
                similarity = self.calculate_enhanced_keypoint_similarity(keypoints, ref["keypoints"])
                improper_similarities.append(similarity)
            
            # Determine best match with confidence levels
            max_proper = max(proper_similarities) if proper_similarities else 0.0
            max_improper = max(improper_similarities) if improper_similarities else 0.0
            
            print(f"Reference comparison - Proper: {max_proper:.2f}, Improper: {max_improper:.2f}")
            
            # High confidence thresholds
            if max_proper > max_improper and max_proper > 0.75:
                return "proper", max_proper
            elif max_improper > max_proper and max_improper > 0.75:
                return "improper", max_improper
            # Medium confidence
            elif max_proper > max_improper and max_proper > 0.65:
                return "proper", max_proper
            elif max_improper > max_proper and max_improper > 0.65:
                return "improper", max_improper
            else:
                return "uncertain", max(max_proper, max_improper)
                
        except Exception as e:
            self.logger.error(f"Reference comparison error: {e}")
            return None, 0.0

    def calculate_enhanced_keypoint_similarity(self, keypoints1, keypoints2):
        """Enhanced similarity calculation focusing on critical joints for each exercise"""
        try:
            # Define critical keypoints for each exercise type
            exercise_critical_points = {
                'squat': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
                'pushup': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip'],
                'plank': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle'],
                'lunge': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
                'deadlift': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']
            }
            
            # Create dictionaries for easier lookup
            kp1_dict = {kp['name']: kp for kp in keypoints1 if kp['confidence'] > 0.5}
            kp2_dict = {kp['name']: kp for kp in keypoints2 if kp['confidence'] > 0.5}
            
            # Use critical points for the specific exercise, fallback to all points
            critical_points = exercise_critical_points.get('squat', [])  # Default to squat
            common_keypoints = set(kp1_dict.keys()) & set(kp2_dict.keys())
            
            # Prioritize critical points in similarity calculation
            critical_common = common_keypoints & set(critical_points)
            other_common = common_keypoints - set(critical_points)
            
            if len(common_keypoints) < 4:  # Need at least 4 common keypoints
                return 0.0
            
            vec1 = []
            vec2 = []
            
            # Add critical points first (weighted more heavily)
            for kp_name in critical_common:
                vec1.extend([kp1_dict[kp_name]['x'], kp1_dict[kp_name]['y']])
                vec2.extend([kp2_dict[kp_name]['x'], kp2_dict[kp_name]['y']])
            
            # Add other points
            for kp_name in other_common:
                vec1.extend([kp1_dict[kp_name]['x'], kp1_dict[kp_name]['y']])
                vec2.extend([kp2_dict[kp_name]['x'], kp2_dict[kp_name]['y']])
            
            if len(vec1) == 0:
                return 0.0
                
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # Calculate weighted similarity (critical points matter more)
            if len(critical_common) > 0:
                # Give more weight to critical points
                critical_weight = 1.5
                other_weight = 1.0
                total_weight = (len(critical_common) * critical_weight + len(other_common) * other_weight)
                
                # Normalize vectors
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                
                # Calculate weighted cosine similarity
                similarity = np.dot(vec1_norm, vec2_norm)
                weighted_similarity = similarity * (len(critical_common) * critical_weight / total_weight)
                
                return max(0.0, weighted_similarity)
            else:
                # Fallback to regular similarity
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                similarity = np.dot(vec1_norm, vec2_norm)
                return max(0.0, similarity)
                
        except Exception as e:
            self.logger.error(f"Enhanced similarity calculation error: {e}")
            return 0.0
    
    def calculate_keypoint_similarity(self, keypoints1, keypoints2):
        """Calculate similarity between two sets of keypoints"""
        try:
            # Create vectors from keypoints (only use confident keypoints)
            vec1 = []
            vec2 = []
            
            # Create dictionaries for easier lookup
            kp1_dict = {kp['name']: kp for kp in keypoints1 if kp['confidence'] > 0.5}
            kp2_dict = {kp['name']: kp for kp in keypoints2 if kp['confidence'] > 0.5}
            
            # Use common keypoints
            common_keypoints = set(kp1_dict.keys()) & set(kp2_dict.keys())
            
            if len(common_keypoints) < 5:  # Need at least 5 common keypoints
                return 0.0
            
            for kp_name in common_keypoints:
                vec1.extend([kp1_dict[kp_name]['x'], kp1_dict[kp_name]['y']])
                vec2.extend([kp2_dict[kp_name]['x'], kp2_dict[kp_name]['y']])
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation error: {e}")
            return 0.0

    # ENHANCED IMAGE-BASED ANALYSIS METHODS (No MediaPipe required)
    def analyze_body_contour(self, image_path, keypoints):
        """Analyze actual body contours and shapes from the image using OpenCV"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Create a binary mask using background subtraction or edge detection
            body_mask = self._create_body_mask(image)
            
            contour_analysis = {}
            
            if body_mask is not None:
                # Analyze body contour from mask
                contour_analysis = self._analyze_body_shape(body_mask, keypoints)
            
            # Analyze back curvature using actual image pixels
            back_analysis = self._analyze_back_curvature(image, keypoints)
            contour_analysis.update(back_analysis)
            
            return contour_analysis
            
        except Exception as e:
            self.logger.error(f"Body contour analysis error: {e}")
            return None
    
    def _create_body_mask(self, image):
        """Create a body mask using background subtraction and edge detection"""
        try:
            # Convert to different color spaces for better segmentation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Apply multiple techniques to create a good mask
            masks = []
            
            # 1. Edge-based mask
            edges = cv2.Canny(gray, 50, 150)
            dilated_edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            masks.append(dilated_edges)
            
            # 2. Threshold-based mask
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(thresh)
            
            # 3. HSV-based mask for skin detection
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            masks.append(skin_mask)
            
            # Combine masks
            combined_mask = np.zeros_like(gray)
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Clean up the mask
            kernel = np.ones((5,5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            return combined_mask
            
        except Exception as e:
            self.logger.error(f"Body mask creation error: {e}")
            return None

    def _extract_back_contour(self, contour, shoulder_mid, hip_mid):
        """Extract back contour points for visualization"""
        try:
            back_points = []
            for point in contour:
                x, y = point[0]
                # Check if point is in back region
                if (shoulder_mid[0] - 50 <= x <= shoulder_mid[0] + 50 and 
                    shoulder_mid[1] <= y <= hip_mid[1]):
                    back_points.append((x, y))
            return back_points
        except Exception as e:
            self.logger.error(f"Back contour extraction error: {e}")
            return []

    def _analyze_back_curvature_enhanced(self, image, keypoints):
        """Enhanced back curvature analysis using actual image contours"""
        analysis = {}
        
        try:
            # Convert keypoints to dictionary
            kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
            
            required_points = ['left_shoulder', 'left_hip', 'right_shoulder', 'right_hip']
            if not all(point in kp_dict for point in required_points):
                return analysis
            
            # Get body mask for contour analysis
            body_mask = self._create_body_mask(image)
            if body_mask is None:
                return analysis
            
            # Find contours
            contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return analysis
            
            # Get largest contour (body)
            body_contour = max(contours, key=cv2.contourArea)
            
            # Extract keypoint positions
            left_shoulder = (int(kp_dict['left_shoulder']['x']), int(kp_dict['left_shoulder']['y']))
            right_shoulder = (int(kp_dict['right_shoulder']['x']), int(kp_dict['right_shoulder']['y']))
            left_hip = (int(kp_dict['left_hip']['x']), int(kp_dict['left_hip']['y']))
            right_hip = (int(kp_dict['right_hip']['x']), int(kp_dict['right_hip']['y']))
            
            # Calculate spine midpoints
            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                        (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_mid = ((left_hip[0] + right_hip[0]) // 2, 
                    (left_hip[1] + right_hip[1]) // 2)
            
            # Analyze back curvature using contour points
            back_curvature = self._calculate_contour_curvature(body_contour, shoulder_mid, hip_mid)
            
            analysis['spine_curvature'] = back_curvature
            
            # Detect arched back
            if back_curvature > self.posture_thresholds['squat']['back_arch_threshold']:
                analysis['arched_back_detected'] = True
                analysis['back_arch_severity'] = back_curvature
            else:
                analysis['arched_back_detected'] = False
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced back curvature analysis error: {e}")
            return analysis

    def _calculate_contour_curvature(self, contour, shoulder_mid, hip_mid):
        """Calculate actual curvature from body contour"""
        try:
            # Extract points along the back region
            back_points = []
            
            for point in contour:
                x, y = point[0]
                # Check if point is in back region (between shoulders and hips)
                if (shoulder_mid[0] - 50 <= x <= shoulder_mid[0] + 50 and 
                    shoulder_mid[1] <= y <= hip_mid[1]):
                    back_points.append((x, y))
            
            if len(back_points) < 10:
                return 0
            
            # Sort points by y-coordinate
            back_points.sort(key=lambda p: p[1])
            
            # Calculate curvature using polynomial fitting
            if len(back_points) > 5:
                y_vals = [p[1] for p in back_points]
                x_vals = [p[0] for p in back_points]
                
                # Fit a polynomial to detect curvature
                degree = 2
                coeffs = np.polyfit(y_vals, x_vals, degree)
                
                # Calculate curvature from second derivative
                if abs(coeffs[0]) > 0.0001:  # Significant curvature
                    curvature = abs(coeffs[0]) * 1000  # Scale for meaningful values
                    return min(curvature, 90)  # Cap at 90 degrees
                    
            return 0
            
        except Exception as e:
            self.logger.error(f"Contour curvature calculation error: {e}")
            return 0

    def compare_with_references_enhanced(self, keypoints, exercise_type):
        """Enhanced reference comparison with better pattern recognition"""
        if exercise_type not in self.reference_positions:
            return None, 0.0
        
        references = self.reference_positions[exercise_type]
        
        # If no references, return uncertain
        if not references["proper"] and not references["improper"]:
            return "uncertain", 0.5  # Default to middle confidence
        
        try:
            # Calculate similarity with proper references
            proper_similarities = []
            for ref in references["proper"]:
                similarity = self.calculate_structural_similarity(keypoints, ref["keypoints"], exercise_type)
                proper_similarities.append(similarity)
            
            # Calculate similarity with improper references
            improper_similarities = []
            for ref in references["improper"]:
                similarity = self.calculate_structural_similarity(keypoints, ref["keypoints"], exercise_type)
                improper_similarities.append(similarity)
            
            print(f"Proper similarities: {proper_similarities}")
            print(f"Improper similarities: {improper_similarities}")
            
            # Use the best match from each category
            best_proper = max(proper_similarities) if proper_similarities else 0.0
            best_improper = max(improper_similarities) if improper_similarities else 0.0
            
            print(f"Best proper: {best_proper:.3f}, Best improper: {best_improper:.3f}")
            
            # More reasonable matching logic
            if best_proper > 0.6 and best_proper > best_improper:
                return "proper", best_proper
            elif best_improper > 0.6 and best_improper > best_proper:
                return "improper", best_improper
            elif best_proper > 0.5:
                return "proper", best_proper
            elif best_improper > 0.5:
                return "improper", best_improper
            else:
                return "uncertain", max(best_proper, best_improper)
                
        except Exception as e:
            self.logger.error(f"Enhanced reference comparison error: {e}")
            return "uncertain", 0.5

    def calculate_structural_similarity(self, keypoints1, keypoints2, exercise_type):
        """Calculate similarity based on structural relationships rather than exact positions"""
        try:
            # Create dictionaries for easier access
            kp1_dict = {kp['name']: kp for kp in keypoints1 if kp['confidence'] > 0.5}
            kp2_dict = {kp['name']: kp for kp in keypoints2 if kp['confidence'] > 0.5}
            
            common_keypoints = set(kp1_dict.keys()) & set(kp2_dict.keys())
            
            if len(common_keypoints) < 4:
                return 0.0
            
            # Calculate similarity based on angular relationships and proportions
            similarity_scores = []
            
            # 1. Torso alignment similarity
            if all(kp in common_keypoints for kp in ['left_shoulder', 'left_hip']):
                torso_angle1 = self._calculate_angle_from_vertical(kp1_dict['left_shoulder'], kp1_dict['left_hip'])
                torso_angle2 = self._calculate_angle_from_vertical(kp2_dict['left_shoulder'], kp2_dict['left_hip'])
                angle_diff = abs(torso_angle1 - torso_angle2)
                torso_similarity = max(0, 1 - angle_diff / 45.0)  # Normalize to 0-1
                similarity_scores.append(torso_similarity)
            
            # 2. Limb proportion similarity
            if all(kp in common_keypoints for kp in ['left_shoulder', 'left_elbow', 'left_wrist']):
                # Arm proportions
                upper_arm1 = self._calculate_distance(kp1_dict['left_shoulder'], kp1_dict['left_elbow'])
                lower_arm1 = self._calculate_distance(kp1_dict['left_elbow'], kp1_dict['left_wrist'])
                arm_ratio1 = upper_arm1 / lower_arm1 if lower_arm1 > 0 else 1.0
                
                upper_arm2 = self._calculate_distance(kp2_dict['left_shoulder'], kp2_dict['left_elbow'])
                lower_arm2 = self._calculate_distance(kp2_dict['left_elbow'], kp2_dict['left_wrist'])
                arm_ratio2 = upper_arm2 / lower_arm2 if lower_arm2 > 0 else 1.0
                
                arm_similarity = 1.0 - min(abs(arm_ratio1 - arm_ratio2) / 0.5, 1.0)
                similarity_scores.append(arm_similarity)
            
            # 3. Keypoint position similarity (normalized)
            position_similarities = []
            for kp_name in common_keypoints:
                kp1 = kp1_dict[kp_name]
                kp2 = kp2_dict[kp_name]
                
                # Normalize positions by body size
                body_height1 = self._estimate_body_height(kp1_dict)
                body_height2 = self._estimate_body_height(kp2_dict)
                
                if body_height1 > 0 and body_height2 > 0:
                    norm_x1 = kp1['x'] / body_height1
                    norm_y1 = kp1['y'] / body_height1
                    norm_x2 = kp2['x'] / body_height2
                    norm_y2 = kp2['y'] / body_height2
                    
                    distance = np.sqrt((norm_x1 - norm_x2)**2 + (norm_y1 - norm_y2)**2)
                    position_similarity = max(0, 1 - distance / 0.5)  # Normalize
                    position_similarities.append(position_similarity)
            
            if position_similarities:
                similarity_scores.append(np.mean(position_similarities))
            
            if not similarity_scores:
                return 0.0
                
            final_similarity = np.mean(similarity_scores)
            return min(1.0, max(0.0, final_similarity))
            
        except Exception as e:
            self.logger.error(f"Structural similarity error: {e}")
            return 0.0

    def _estimate_body_height(self, kp_dict):
        """Estimate body height from keypoints for normalization"""
        try:
            if 'left_shoulder' in kp_dict and 'left_ankle' in kp_dict:
                return abs(kp_dict['left_ankle']['y'] - kp_dict['left_shoulder']['y'])
            elif 'nose' in kp_dict and 'left_ankle' in kp_dict:
                return abs(kp_dict['left_ankle']['y'] - kp_dict['nose']['y'])
            else:
                return 400.0  # Default estimate
        except:
            return 400.0

    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

    def calculate_exercise_specific_similarity(self, keypoints1, keypoints2, exercise_type):
        """Exercise-specific similarity calculation focusing on critical joints"""
        try:
            # Define critical keypoints and their weights for each exercise
            exercise_weights = {
                'squat': {
                    'left_hip': 2.0, 'right_hip': 2.0,
                    'left_knee': 1.5, 'right_knee': 1.5,
                    'left_ankle': 1.0, 'right_ankle': 1.0,
                    'left_shoulder': 0.5, 'right_shoulder': 0.5
                },
                'pushup': {
                    'left_shoulder': 2.0, 'right_shoulder': 2.0,
                    'left_elbow': 1.5, 'right_elbow': 1.5,
                    'left_hip': 1.0, 'right_hip': 1.0,
                    'left_wrist': 0.5, 'right_wrist': 0.5
                },
                'plank': {
                    'left_shoulder': 1.5, 'right_shoulder': 1.5,
                    'left_hip': 2.0, 'right_hip': 2.0,
                    'left_ankle': 1.0, 'right_ankle': 1.0
                }
            }
            
            weights = exercise_weights.get(exercise_type, {})
            
            # Create dictionaries for easier access
            kp1_dict = {kp['name']: kp for kp in keypoints1 if kp['confidence'] > 0.5}
            kp2_dict = {kp['name']: kp for kp in keypoints2 if kp['confidence'] > 0.5}
            
            common_keypoints = set(kp1_dict.keys()) & set(kp2_dict.keys())
            
            if len(common_keypoints) < 4:
                return 0.0
            
            total_similarity = 0.0
            total_weight = 0.0
            
            for kp_name in common_keypoints:
                weight = weights.get(kp_name, 1.0)
                
                kp1 = kp1_dict[kp_name]
                kp2 = kp2_dict[kp_name]
                
                # Calculate distance-based similarity
                distance = np.sqrt((kp1['x'] - kp2['x'])**2 + (kp1['y'] - kp2['y'])**2)
                
                # Normalize distance (assuming image size ~1000px)
                normalized_distance = min(distance / 200.0, 1.0)
                similarity = 1.0 - normalized_distance
                
                total_similarity += similarity * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
                
            return total_similarity / total_weight
            
        except Exception as e:
            self.logger.error(f"Exercise-specific similarity error: {e}")
            return 0.0

    def draw_posture_feedback(self, image, keypoints, issues, status="needs_work"):
        """Draw visual feedback on the image with color coding based on status"""
        try:
            # Convert keypoints to dictionary
            kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
            
            # Set colors based on status
            if status == "proper" or status == "good":
                # GREEN for good form
                line_color = (0, 255, 0)  # Green
                text_color = (0, 180, 0)  # Darker green for better visibility
                draw_ideal_lines = False  # Don't show ideal lines for good form
            else:
                # RED/YELLOW for needs improvement
                if status == "needs_work":
                    line_color = (255, 165, 0)  # Orange
                    text_color = (200, 100, 0)  # Dark orange
                else:  # improper
                    line_color = (0, 0, 255)  # Red
                    text_color = (180, 0, 0)  # Dark red
                draw_ideal_lines = True  # Show ideal lines for guidance
            
            # Draw spine alignment indicator
            if all(kp in kp_dict for kp in ['left_shoulder', 'left_hip', 'right_shoulder', 'right_hip']):
                self._draw_spine_indicator(image, kp_dict, line_color, text_color, status, draw_ideal_lines)
            
            # Draw knee alignment indicators if there are knee issues OR if we're showing good form
            if any("knee" in issue.lower() for issue in issues) or status in ["proper", "good"]:
                self._draw_knee_alignment_indicator(image, kp_dict, line_color, text_color, status, draw_ideal_lines)
                    
            # Add success elements for good form
            if status in ["proper", "good"]:
                self._draw_success_elements(image, kp_dict)
                    
            return image
            
        except Exception as e:
            self.logger.error(f"Posture feedback drawing error: {e}")
            return image

    def _draw_spine_indicator(self, image, kp_dict, line_color, text_color, status, draw_ideal_lines=True):
        """Draw spine alignment indicator with appropriate color - NO TEXT"""
        try:
            left_shoulder = (int(kp_dict['left_shoulder']['x']), int(kp_dict['left_shoulder']['y']))
            left_hip = (int(kp_dict['left_hip']['x']), int(kp_dict['left_hip']['y']))
            right_shoulder = (int(kp_dict['right_shoulder']['x']), int(kp_dict['right_shoulder']['y']))
            right_hip = (int(kp_dict['right_hip']['x']), int(kp_dict['right_hip']['y']))
            
            # Calculate midpoints
            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                        (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_mid = ((left_hip[0] + right_hip[0]) // 2, 
                    (left_hip[1] + right_hip[1]) // 2)
            
            # Draw the actual spine line
            cv2.line(image, shoulder_mid, hip_mid, line_color, 4)
            
            # For good form, draw visual indicators (NO TEXT)
            if status in ["proper", "good"]:
                # Draw green checkmark using lines instead of text
                check_x = shoulder_mid[0] - 80
                check_y = shoulder_mid[1] - 30
                
                # Draw checkmark using lines (no text characters)
                cv2.line(image, (check_x - 8, check_y), (check_x - 2, check_y + 8), text_color, 3)
                cv2.line(image, (check_x - 2, check_y + 8), (check_x + 10, check_y - 6), text_color, 3)
                    
                # Draw sparkles around the checkmark
                self._draw_sparkles(image, shoulder_mid[0] - 100, shoulder_mid[1] - 50)
            else:
                # For needs improvement, draw the ideal spine line for comparison (in light gray)
                if draw_ideal_lines:
                    ideal_hip = (shoulder_mid[0], hip_mid[1])  # Directly below shoulder
                    cv2.line(image, shoulder_mid, ideal_hip, (200, 200, 200), 2)
            
            # REMOVED ALL TEXT DRAWING OPERATIONS
            # The feedback will be shown in the chatbot messages, not on the image
                            
        except Exception as e:
            self.logger.error(f"Spine indicator drawing error: {e}")

    def _draw_knee_alignment_indicator(self, image, kp_dict, line_color, text_color, status, draw_ideal_lines=True):
        """Draw knee alignment feedback with appropriate color - NO TEXT"""
        try:
            if all(kp in kp_dict for kp in ['left_knee', 'left_ankle']):
                knee = (int(kp_dict['left_knee']['x']), int(kp_dict['left_knee']['y']))
                ankle = (int(kp_dict['left_ankle']['x']), int(kp_dict['left_ankle']['y']))
                
                # Draw line from knee to ankle
                cv2.line(image, knee, ankle, line_color, 3)
                
                if status not in ["proper", "good"] and draw_ideal_lines:
                    # For needs improvement, show ideal alignment (light gray)
                    ideal_ankle = (knee[0], ankle[1])  # Ankle directly below knee
                    cv2.line(image, knee, ideal_ankle, (200, 200, 200), 2)
                
                # REMOVED TEXT - no more "Knee" text that could show as "???"
                            
        except Exception as e:
            self.logger.error(f"Knee alignment indicator error: {e}")

    def _draw_success_elements(self, image, kp_dict):
        """Draw celebration elements for excellent form - NO TEXT"""
        try:
            if 'left_shoulder' in kp_dict:
                shoulder = (int(kp_dict['left_shoulder']['x']), int(kp_dict['left_shoulder']['y']))
                
                # Draw celebration elements without text
                # Draw a circle with plus signs to indicate success
                center_x = shoulder[0] - 50
                center_y = shoulder[1] - 100
                
                # Draw a celebration circle
                cv2.circle(image, (center_x, center_y), 25, (0, 180, 0), 3)
                
                # Draw plus signs inside the circle
                cv2.line(image, (center_x - 8, center_y), (center_x + 8, center_y), (0, 180, 0), 3)
                cv2.line(image, (center_x, center_y - 8), (center_x, center_y + 8), (0, 180, 0), 3)
                            
        except Exception as e:
            self.logger.error(f"Success elements drawing error: {e}")

    def _draw_sparkles(self, image, center_x, center_y):
        """Draw sparkle effects around a point"""
        try:
            import random
            # Draw multiple small circles as sparkles
            for i in range(8):
                angle = random.uniform(0, 2 * 3.14159)
                distance = random.randint(20, 40)
                x = int(center_x + distance * math.cos(angle))
                y = int(center_y + distance * math.sin(angle))
                
                # Random size and brightness
                size = random.randint(2, 4)
                brightness = random.randint(200, 255)
                cv2.circle(image, (x, y), size, (brightness, brightness, 0), -1)
        except Exception as e:
            self.logger.error(f"Sparkles drawing error: {e}")
                
    def _analyze_back_curvature(self, image, keypoints):
        """Analyze back curvature using actual image pixels and keypoints"""
        analysis = {}
        
        try:
            # Convert keypoints to dictionary
            kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
            
            required_points = ['left_shoulder', 'left_hip', 'right_shoulder', 'right_hip']
            if not all(point in kp_dict for point in required_points):
                return analysis
            
            # Calculate spine line using midpoints
            left_shoulder = (int(kp_dict['left_shoulder']['x']), int(kp_dict['left_shoulder']['y']))
            left_hip = (int(kp_dict['left_hip']['x']), int(kp_dict['left_hip']['y']))
            right_shoulder = (int(kp_dict['right_shoulder']['x']), int(kp_dict['right_shoulder']['y']))
            right_hip = (int(kp_dict['right_hip']['x']), int(kp_dict['right_hip']['y']))
            
            # Calculate midpoints for spine
            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )
            hip_mid = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2
            )
            
            # Calculate spine curvature using keypoint-based approach
            spine_curvature = self._calculate_spine_curvature_from_keypoints(
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            
            analysis['spine_curvature'] = spine_curvature
            
            # Detect arched back
            if spine_curvature > self.posture_thresholds['squat']['back_arch_threshold']:
                analysis['arched_back_detected'] = True
                analysis['back_arch_severity'] = spine_curvature
            else:
                analysis['arched_back_detected'] = False
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Back curvature analysis error: {e}")
            return analysis
    
    def _calculate_spine_curvature_from_keypoints(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """Calculate spine curvature using keypoint positions"""
        try:
            # Calculate midpoints
            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )
            hip_mid = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2
            )
            
            # Calculate the angle of the spine relative to vertical
            dx = hip_mid[0] - shoulder_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]
            
            if dy == 0:
                return 90
            
            # Calculate angle from vertical (0° = perfectly vertical)
            angle_from_vertical = abs(math.degrees(math.atan2(dx, dy)))
            
            # For squat analysis, we want to detect excessive forward lean
            # A larger angle means more forward lean/arched back
            return angle_from_vertical
            
        except Exception as e:
            self.logger.error(f"Spine curvature calculation error: {e}")
            return 0
    
    def _analyze_body_shape(self, body_mask, keypoints):
        """Analyze overall body shape and alignment"""
        analysis = {}
        
        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return analysis
            
            # Get the largest contour (body)
            body_contour = max(contours, key=cv2.contourArea)
            
            # Calculate body bounding rectangle
            rect = cv2.minAreaRect(body_contour)
            box = cv2.boxPoints(rect)
            # FIX: Replace np.int0 with np.int32 or np.int_
            box = np.int32(box)  # Changed from np.int0
            
            # Calculate aspect ratio and orientation
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = width / height if height > 0 else 0
            orientation = rect[2]
            
            analysis['body_aspect_ratio'] = aspect_ratio
            analysis['body_orientation'] = orientation
            
            # Detect hunched posture based on aspect ratio and orientation
            if aspect_ratio > 0.8 and abs(orientation) > 45:
                analysis['hunched_posture_detected'] = True
            else:
                analysis['hunched_posture_detected'] = False
            
            # Additional analysis using convexity defects to detect rounded back
            hull = cv2.convexHull(body_contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(body_contour, hull)
                if defects is not None:
                    # Count significant convexity defects (potential rounded back)
                    significant_defects = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 1000:  # Threshold for significant defect
                            significant_defects += 1
                    
                    if significant_defects > 2:
                        analysis['rounded_back_detected'] = True
                    else:
                        analysis['rounded_back_detected'] = False
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Body shape analysis error: {e}")
            return analysis

    def _combine_analyses(self, traditional_issues, traditional_suggestions, contour_analysis, exercise_type):
        """Combine traditional keypoint analysis with image-based contour analysis"""
        issues = traditional_issues.copy()
        suggestions = traditional_suggestions.copy()
        
        if contour_analysis is None:
            return issues, suggestions
        
        try:
            # Add image-based analysis findings
            if contour_analysis.get('arched_back_detected', False):
                issues.append("Arched back detected from image analysis")
                suggestions.append("Focus on maintaining a neutral spine throughout the movement")
            
            if contour_analysis.get('hunched_posture_detected', False):
                issues.append("Hunched posture detected")
                suggestions.append("Keep shoulders back and chest up")
            
            if contour_analysis.get('rounded_back_detected', False):
                issues.append("Rounded back detected")
                suggestions.append("Engage core muscles to maintain proper spinal alignment")
            
            # Exercise-specific contour analysis
            if exercise_type == 'squat':
                spine_curvature = contour_analysis.get('spine_curvature', 0)
                if spine_curvature > self.posture_thresholds['squat']['back_arch_threshold']:
                    if "Arched back" not in " ".join(issues):
                        issues.append(f"Excessive spine curvature ({spine_curvature:.1f}°)")
                        suggestions.append("Maintain a more neutral spine position")
            
            elif exercise_type == 'pushup':
                body_ratio = contour_analysis.get('body_aspect_ratio', 0)
                if body_ratio > 0.8:
                    issues.append("Body alignment needs improvement")
                    suggestions.append("Keep your body in a straight line from head to heels")
            
        except Exception as e:
            self.logger.error(f"Error combining analyses: {e}")
        
        return issues, suggestions
        
    def enhanced_analyze_exercise(self, keypoints, image_path, user_selected_exercise=None):
        """Enhanced analysis with better balance between detection methods"""
        if keypoints is None or len(keypoints) == 0:
            return (["No pose detected"], ["Try to be more visible in the frame"]), 'general'
        
        # Determine exercise type
        if user_selected_exercise and user_selected_exercise != 'general':
            exercise_type = user_selected_exercise
            print(f"Using user-selected exercise: {exercise_type}")
        else:
            exercise_type = self.detect_exercise_yolov11(image_path)
            print(f"Auto-detected exercise: {exercise_type}")
        
        # Read image for contour analysis
        image = cv2.imread(image_path)
        if image is None:
            return (["Could not read image"], ["Please try a different image"]), exercise_type
        
        # STEP 1: Enhanced reference comparison
        reference_match, similarity_score = self.compare_with_references_enhanced(keypoints, exercise_type)
        print(f"Reference comparison - Match: {reference_match}, Score: {similarity_score:.3f}")
        
        # STEP 2: Traditional keypoint analysis
        traditional_issues, traditional_suggestions = self._get_traditional_analysis(keypoints, exercise_type)
        
        # STEP 3: Contour analysis for back arch
        contour_analysis = self._analyze_back_curvature_enhanced(image, keypoints)
        
        # STEP 4: Combine analyses with more balanced approach
        final_issues, final_suggestions = self._combine_analyses_balanced(
            traditional_issues, traditional_suggestions, 
            contour_analysis, exercise_type,
            reference_match, similarity_score
        )
        
        # BE MORE GENEROUS - if reference says proper and similarity is decent, reduce issues
        if reference_match == "proper" and similarity_score > 0.6:
            if len(final_issues) <= 2:  # Only minor issues
                final_issues = []  # Clear minor issues for good reference match
                final_suggestions = ["Excellent form! Your technique matches proper examples well"]
            elif len(final_issues) > 2:
                final_issues = final_issues[:1]  # Keep only the most critical issue
                final_suggestions = ["Good form with one area for refinement"]
        
        print(f"Final issues: {final_issues}")
        return (final_issues, final_suggestions), exercise_type

    def _combine_analyses_balanced(self, traditional_issues, traditional_suggestions, 
                                contour_analysis, exercise_type,
                                reference_match, similarity_score):
        """More balanced combination of analysis methods"""
        issues = traditional_issues.copy()
        suggestions = traditional_suggestions.copy()
        
        # Add contour analysis findings if confident
        if contour_analysis and contour_analysis.get('arched_back_detected', False):
            severity = contour_analysis.get('back_arch_severity', 0)
            if severity > 25:  # Only add if significant
                issues.append(f"Arched back detected (curvature: {severity:.1f}°)")
                suggestions.append("Focus on maintaining a neutral spine throughout the movement")
        
        # If reference strongly suggests proper form, be more conservative with issues
        if reference_match == "proper" and similarity_score > 0.7:
            # Only keep the most critical issues
            if len(issues) > 1:
                critical_issues = [issue for issue in issues if any(keyword in issue.lower() 
                                for keyword in ['arch', 'rounded', 'strain', 'injur'])]
                if critical_issues:
                    issues = critical_issues[:1]  # Keep only one critical issue
                else:
                    issues = issues[:1]  # Keep only the first issue
        
        return issues, suggestions

    def _combine_all_analyses(self, reference_match, similarity_score, 
                            pose_issues, pose_suggestions, 
                            contour_analysis, exercise_type):
        """Intelligently combine all analysis methods"""
        
        issues = pose_issues.copy()
        suggestions = pose_suggestions.copy()
        
        # Add contour analysis findings
        if contour_analysis.get('arched_back_detected', False):
            severity = contour_analysis.get('back_arch_severity', 0)
            issues.append(f"Arched back detected (curvature: {severity:.1f}°)")
            suggestions.append("Focus on maintaining a neutral spine - engage your core")
        
        # Use reference data to override minor issues if similarity is high
        if reference_match == "proper" and similarity_score > 0.7:
            if len(issues) <= 2:  # Only minor issues
                # Keep the analysis but acknowledge good overall form
                if issues:
                    suggestions.append("Overall form matches proper examples well")
            elif len(issues) > 2 and similarity_score > 0.8:
                # Strong reference match might indicate false positives in pose analysis
                issues = issues[:2]  # Keep only the most critical issues
                suggestions.append("Good foundation with some areas for refinement")
        
        return issues, suggestions

    def _combine_reference_and_pose_analysis(self, reference_match, similarity_score, pose_issues, pose_suggestions, exercise_type):
        """Intelligently combine reference-based and pose-based analysis"""
        
        # If we have a strong reference match, trust it more
        if reference_match == "proper" and similarity_score > 0.8:
            print(f"Strong proper reference match: {similarity_score:.2f}")
            # Even if pose analysis found minor issues, trust the reference
            if len(pose_issues) <= 1:  # Only minor pose issues
                return [], ["Excellent form! Matches proper reference examples"]
            else:
                # Keep major pose issues but acknowledge good reference match
                return pose_issues, pose_suggestions + ["Overall form matches proper examples well"]
        
        elif reference_match == "improper" and similarity_score > 0.8:
            print(f"Strong improper reference match: {similarity_score:.2f}")
            # Trust the improper reference classification
            major_issues = ["Form matches improper reference examples"] + pose_issues
            return major_issues, pose_suggestions + ["Focus on correcting fundamental form issues"]
        
        elif reference_match == "proper" and similarity_score > 0.7:
            print(f"Moderate proper reference match: {similarity_score:.2f}")
            # Moderate confidence - combine with pose analysis
            if len(pose_issues) == 0:
                return [], ["Good form! Similar to proper references"]
            else:
                return pose_issues, pose_suggestions + ["Overall form is good with some adjustments needed"]
        
        else:
            # No strong reference match - rely more on pose analysis
            print(f"Weak or no reference match: {reference_match}, score: {similarity_score:.2f}")
            
            # If we have some reference data but weak match, use it as context
            if reference_match == "proper" and similarity_score > 0.6:
                if len(pose_issues) == 0:
                    return [], ["Form looks good based on pose analysis"]
                else:
                    return pose_issues, pose_suggestions + ["Some similarity to proper form examples"]
            
            elif reference_match == "improper" and similarity_score > 0.6:
                if len(pose_issues) > 0:
                    return ["Form shows issues similar to improper examples"] + pose_issues, pose_suggestions
                else:
                    return ["Caution: Form similar to improper examples"], ["Review proper technique fundamentals"]
            
            else:
                # No useful reference data - rely entirely on pose analysis
                return pose_issues, pose_suggestions
    
    def _get_traditional_analysis(self, keypoints, exercise_type):
        """Get traditional keypoint-based analysis"""
        if exercise_type == 'squat':
            return self.analyze_squat(keypoints)
        elif exercise_type == 'pushup':
            return self.analyze_pushup(keypoints)
        elif exercise_type == 'plank':
            return self.analyze_plank(keypoints)
        elif exercise_type == 'lunge':
            return self.analyze_lunge(keypoints)
        elif exercise_type == 'deadlift':
            return self.analyze_deadlift(keypoints)
        else:
            return self.analyze_general(keypoints)

    # ORIGINAL ANALYSIS METHODS (keep these for backward compatibility)
    def analyze_exercise(self, keypoints, image_path, user_selected_exercise=None):
        """Enhanced analysis with reference comparison and image-based analysis"""
        if keypoints is None or len(keypoints) == 0:
            return (["No pose detected"], ["Try to be more visible in the frame"]), 'general'
        
        # Use user-selected exercise type if provided, otherwise auto-detect
        if user_selected_exercise and user_selected_exercise != 'general':
            exercise_type = user_selected_exercise
            print(f"Using user-selected exercise: {exercise_type}")
        else:
            # Auto-detect only if user didn't specify or selected 'general'
            exercise_type = self.detect_exercise_yolov11(image_path)
            print(f"Auto-detected exercise: {exercise_type}")
        
        # Analyze body contours and actual image content
        contour_analysis = self.analyze_body_contour(image_path, keypoints)
        
        # Get traditional keypoint analysis
        traditional_issues, traditional_suggestions = self._get_traditional_analysis(keypoints, exercise_type)
        
        # Combine with image-based analysis
        enhanced_issues, enhanced_suggestions = self._combine_analyses(
            traditional_issues, traditional_suggestions, contour_analysis, exercise_type
        )
        
        return (enhanced_issues, enhanced_suggestions), exercise_type     

    def analyze_squat(self, keypoints):
        """Analyze squat form using keypoints"""
        issues = []
        suggestions = []
        
        if not keypoints:
            return ["No pose detected"], ["Ensure full body is visible in frame"]
        
        # Convert to dictionary for easier access
        kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
        
        # Check if we have necessary keypoints
        required_points = ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']
        if not all(key in kp_dict for key in required_points):
            return ["Insufficient pose detection"], ["Make sure your full body is visible"]
        
        # Analyze forward lean
        shoulder = kp_dict['left_shoulder']
        hip = kp_dict['left_hip']
        torso_angle = self._calculate_angle_from_vertical(shoulder, hip)
        
        if torso_angle > 30:
            issues.append(f"Excessive forward lean ({int(torso_angle)}°)")
            suggestions.append("Keep chest up and maintain more upright torso")
        
        # Check knee position
        knee = kp_dict['left_knee']
        ankle = kp_dict['left_ankle']
        if knee['x'] > ankle['x'] + 50:  # Knee too far forward
            issues.append("Knees extending beyond toes")
            suggestions.append("Initiate movement by sitting hips back first")
        
        return issues, suggestions
    
    def analyze_pushup(self, keypoints):
        """Analyze push-up form using keypoints"""
        issues = []
        suggestions = []
        
        if not keypoints:
            return ["No pose detected"], ["Ensure full body is visible in frame"]
        
        kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
        
        required_points = ['left_shoulder', 'left_hip', 'left_ankle']
        if not all(key in kp_dict for key in required_points):
            return ["Insufficient pose detection"], ["Make sure your full body is visible"]
        
        shoulder = kp_dict['left_shoulder']
        hip = kp_dict['left_hip']
        ankle = kp_dict['left_ankle']
        
        # Check body alignment
        if hip['y'] < shoulder['y'] - 50:
            issues.append("Hips too high (mountain butt)")
            suggestions.append("Lower your hips to align with shoulders")
        elif hip['y'] > ankle['y'] + 50:
            issues.append("Hips sagging (arched back)")
            suggestions.append("Engage your core to maintain straight line")
        
        return issues, suggestions
    
    def analyze_plank(self, keypoints):
        """Analyze plank form"""
        return self.analyze_pushup(keypoints)  # Similar analysis
    
    def analyze_lunge(self, keypoints):
        """Analyze lunge form"""
        issues = []
        suggestions = []
        
        if not keypoints:
            return ["No pose detected"], ["Ensure full body is visible in frame"]
        
        # Basic lunge analysis
        kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
        
        if 'left_knee' in kp_dict and 'left_ankle' in kp_dict:
            knee = kp_dict['left_knee']
            ankle = kp_dict['left_ankle']
            
            if knee['x'] > ankle['x'] + 50:
                issues.append("Front knee extending too far forward")
                suggestions.append("Keep front knee aligned with ankle")
        
        return issues, suggestions
    
    def analyze_deadlift(self, keypoints):
        """Analyze deadlift form"""
        issues = []
        suggestions = []
        
        if not keypoints:
            return ["No pose detected"], ["Ensure full body is visible in frame"]
        
        kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
        
        if 'left_shoulder' in kp_dict and 'left_hip' in kp_dict:
            shoulder = kp_dict['left_shoulder']
            hip = kp_dict['left_hip']
            
            back_angle = self._calculate_angle_from_vertical(shoulder, hip)
            if back_angle > 45:
                issues.append("Back too rounded")
                suggestions.append("Keep chest up and maintain neutral spine")
        
        return issues, suggestions
    
    def analyze_general(self, keypoints):
        """General posture analysis"""
        issues = []
        suggestions = []
        
        if not keypoints:
            return ["No pose detected"], ["Ensure full body is visible"]
        
        # Basic posture checks
        kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.5}
        
        if 'left_shoulder' in kp_dict and 'left_hip' in kp_dict:
            shoulder = kp_dict['left_shoulder']
            hip = kp_dict['left_hip']
            
            if shoulder['y'] > hip['y'] + 50:
                issues.append("Forward shoulder posture detected")
                suggestions.append("Try to relax shoulders back")
        
        return issues, suggestions
    
    def _calculate_angle_from_vertical(self, upper_point, lower_point):
        """Calculate angle relative to vertical"""
        dx = lower_point['x'] - upper_point['x']
        dy = lower_point['y'] - upper_point['y']
        
        if dy == 0:
            return 90
        
        angle = np.degrees(np.arctan(abs(dx) / abs(dy)))
        return angle
    
    def get_training_statistics(self):
        """Get statistics about training data"""
        stats = {}
        
        for exercise in self.exercise_classes:
            # Get counts from the actual reference_positions data structure
            proper_count = len(self.reference_positions.get(exercise, {}).get("proper", []))
            improper_count = len(self.reference_positions.get(exercise, {}).get("improper", []))
            
            stats[exercise] = {
                "proper_references": proper_count,
                "improper_references": improper_count,
                "total_references": proper_count + improper_count
            }
        
        return stats