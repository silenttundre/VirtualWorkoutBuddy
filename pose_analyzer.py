import cv2
import numpy as np
import torch
import mediapipe as mp
from torchvision import transforms
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import sys
sys.path.append('yolov5')

class PoseAnalyzer:
    def __init__(self):
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MediaPipe Pose with proper configuration
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,  # Disable segmentation if not needed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize YOLO model
        self.yolo_model = self.load_yolo_model()
        self.exercise_classes = ['squat', 'pushup', 'plank', 'lunge', 'deadlift']
    
    def load_yolo_model(self):
        """Load YOLOv5 model with error handling"""
        try:
            model = attempt_load('yolov5s.pt', device=self.device)
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    
    def detect_pose(self, image_path):
        """Detect human pose using MediaPipe with proper image handling"""
        try:
            # Read image with explicit dimensions
            image = cv2.imread(image_path)
            if image is None:
                return None, None, "Could not read image"
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            # Draw pose landmarks
            annotated_image = image.copy()
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS)
            
            return results.pose_landmarks, annotated_image, None
        except Exception as e:
            return None, None, str(e)
    
    def analyze_exercise(self, landmarks, image):
        """Improved analysis with better good/bad determination"""
        if landmarks is None:
            return (["No pose detected"], ["Try to be more visible in the frame"]), 'general'
        
        exercise_type = self.detect_exercise(image)
        
        # Get the specific analysis
        if exercise_type == 'squat':
            issues, suggestions = self.analyze_squat(landmarks)
        elif exercise_type == 'pushup':
            issues, suggestions = self.analyze_pushup(landmarks)
        # ... other exercises ...
        else:
            issues, suggestions = self.analyze_general(landmarks)
        
        # Only flag as bad if there are significant issues
        if len(issues) > 2 or any("significant" in issue.lower() for issue in issues):
            assessment = "Needs improvement"
        elif len(issues) > 0:
            assessment = "Generally good with minor notes"
        else:
            assessment = "Good form!"
        
        return (issues, suggestions), exercise_type
        
    def analyze_general(self, landmarks):
        """More balanced general posture analysis"""
        issues = []
        suggestions = []
        
        if not landmarks:
            return ["No pose detected"], ["Ensure full body is visible"]
        
        # Shoulder position check with tolerance
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        if left_shoulder.y > left_hip.y + 0.1:  # More tolerant threshold
            issues.append("Slight forward shoulder posture")
            suggestions.append("Try to relax shoulders back slightly")
        
        # Head position with tolerance
        ear_pos = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        shoulder_pos = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        if ear_pos.x > shoulder_pos.x + 0.1:  # Increased threshold
            issues.append("Mild forward head posture")
            suggestions.append("Bring head back over shoulders")
        
        # Only return issues if we found significant problems
        if len(issues) >= 2:  # Require multiple issues to flag
            return issues, suggestions
        return [], []  # Return empty if minor or no issues
        
    def _is_confidence_high(self, landmarks, *joint_indices):
        """Check if landmark detection confidence is high enough"""
        CONFIDENCE_THRESHOLD = 0.7
        for idx in joint_indices:
            if landmarks.landmark[idx].visibility < CONFIDENCE_THRESHOLD:
                return False
        return True

    def analyze_squat(self, landmarks):
        """Detect and correct bad squat form with forward lean"""
        issues = []
        suggestions = []
        warnings = []
        
        if not landmarks:
            return (["No pose detected"], ["Ensure full body is visible in frame"]), []

        # Get key landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        # 1. Detect excessive forward lean (chest down)
        # Calculate torso angle (shoulder to hip relative to vertical)
        torso_angle = self._calculate_vertical_angle(
            left_shoulder, left_hip
        )
        
        if torso_angle > 30:  # More than 30 degrees from vertical
            issues.append(f"Excessive forward lean ({int(torso_angle)}°)")
            suggestions.append("Keep chest up and maintain more upright torso")
            warnings.append("Forward lean increases back strain")

        # 2. Detect arched spine (improper spinal alignment)
        mid_shoulder = (left_shoulder.y + right_shoulder.y) / 2
        if nose.y > mid_shoulder + 0.15:  # Head significantly forward
            issues.append("Arched spine/forward head posture")
            suggestions.append("Maintain neutral spine - imagine a straight line from head to tailbone")
            warnings.append("Poor spinal alignment can lead to injury")

        # 3. Check knee position relative to feet
        knee_over_toe = left_knee.x - left_ankle.x  # Positive = knees forward of toes
        if abs(knee_over_toe) > 0.08:  # Knees too far forward
            issues.append(f"Knees extending beyond toes ({'forward' if knee_over_toe > 0 else 'backward'})")
            suggestions.append("Initiate movement by sitting hips back first")
            warnings.append("Excessive knee travel increases joint stress")

        # 4. Detect knee valgus (legs collapsing inward)
        left_knee_pos = left_knee.x
        right_knee_pos = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x
        left_ankle_pos = left_ankle.x
        right_ankle_pos = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x
        
        valgus_threshold = 0.06
        left_valgus = left_knee_pos - left_ankle_pos
        right_valgus = right_knee_pos - right_ankle_pos
        
        if left_valgus < -valgus_threshold or right_valgus > valgus_threshold:
            issues.append("Knees collapsing inward (valgus)")
            suggestions.append("Push knees outward to align with second toe")
            warnings.append("Knee valgus increases ACL injury risk")

        # 5. Depth check (only if other form is acceptable)
        if len(issues) == 0:
            knee_angle = self._calculate_joint_angle(
                landmarks,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE
            )
            if knee_angle > 140:
                issues.append("Insufficient depth")
                suggestions.append("Aim for thighs parallel to ground")
            elif knee_angle < 70:
                issues.append("Excessive depth")
                suggestions.append("Maintain control through full range")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "warnings": warnings,
            "metrics": {
                "torso_angle": torso_angle,
                "knee_over_toe": knee_over_toe,
                "knee_valgus": (left_valgus, right_valgus)
            }
        }

    def _calculate_vertical_angle(self, upper_point, lower_point):
        """Calculate angle relative to vertical (0° = perfectly upright)"""
        vertical = np.array([0, -1])  # Negative Y is up in image coordinates
        torso_vector = np.array([lower_point.x - upper_point.x, 
                            lower_point.y - upper_point.y])
        
        # Normalize vectors
        vertical = vertical / np.linalg.norm(vertical)
        torso_vector = torso_vector / np.linalg.norm(torso_vector)
        
        # Calculate angle in degrees
        angle = np.degrees(np.arccos(np.clip(np.dot(vertical, torso_vector), -1.0, 1.0)))
        return angle

    def analyze_pushup(self, landmarks):
        """Analyze push-up form with emphasis on straight-line body position"""
        analysis = {
            'is_proper_form': True,
            'main_issue': None,
            'specific_issues': [],
            'corrections': [],
            'form_score': 100  # Start with perfect score
        }

        if not landmarks:
            return {
                'is_proper_form': False,
                'main_issue': "No pose detected",
                'corrections': ["Ensure full body is visible in frame"]
            }

        # Key landmarks for push-up analysis
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]

        # Calculate mid-points for symmetry
        mid_shoulder = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip = (left_hip.y + right_hip.y) / 2

        # 1. Check for straight line from head to ankles
        body_angle = self._calculate_body_angle(
            nose, 
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR],
            left_shoulder,
            left_hip,
            left_ankle
        )

        # 2. Detect "mountain butt" (hips too high)
        hip_ratio = (mid_hip - mid_shoulder) / (left_ankle.y - mid_shoulder)
        if hip_ratio < -0.15:  # Hips significantly higher than shoulders
            analysis['is_proper_form'] = False
            analysis['main_issue'] = "Hips raised too high (mountain butt)"
            analysis['specific_issues'].append("Butt sticking up in the air")
            analysis['corrections'].append("Lower your hips to align with shoulders")
            analysis['form_score'] -= 30

        # 3. Detect arched back (sagging hips)
        elif hip_ratio > 0.15:  # Hips significantly lower than shoulders
            analysis['is_proper_form'] = False
            analysis['main_issue'] = "Arched back (hips sagging)"
            analysis['specific_issues'].append("Lower back is collapsing")
            analysis['corrections'].append("Engage your core to maintain straight line")
            analysis['form_score'] -= 30

        # 4. Check overall body alignment angle
        if abs(body_angle) > 10:  # More than 10 degrees from horizontal
            analysis['is_proper_form'] = False
            analysis['specific_issues'].append(f"Body alignment is {int(body_angle)}° from straight")
            analysis['corrections'].append("Maintain straight line from head to ankles")
            analysis['form_score'] -= 20

        # 5. Final assessment
        if analysis['form_score'] >= 90:
            analysis['feedback'] = "Excellent push-up form! Maintain that straight body line."
        elif analysis['form_score'] >= 70:
            analysis['feedback'] = "Good form with minor adjustments needed."
        else:
            analysis['feedback'] = "Needs significant improvement - focus on body alignment."

        return analysis

    def _calculate_body_angle(self, nose, ear, shoulder, hip, ankle):
        """Calculate deviation from straight line (0° = perfect horizontal line)"""
        # Vector from shoulder to hip (torso)
        torso_vec = np.array([hip.x - shoulder.x, hip.y - shoulder.y])
        # Vector from hip to ankle (legs)
        legs_vec = np.array([ankle.x - hip.x, ankle.y - hip.y])
        
        # Combined body vector
        body_vec = torso_vec + legs_vec
        
        # Calculate angle relative to horizontal (in image coordinates)
        horizontal = np.array([1, 0])  # Positive x-direction
        angle = np.degrees(np.arccos(
            np.dot(body_vec, horizontal) / 
            (np.linalg.norm(body_vec) * np.linalg.norm(horizontal))))
        
        # Determine if angle is upward or downward
        if body_vec[1] < 0:  # Negative y is upward in image coords
            return angle  # Positive angle (butt up)
        else:
            return -angle  # Negative angle (hips sagging)
    
    # Add similar methods for other exercises (plank, lunge, deadlift)
    def analyze_plank(self, landmarks):
        """Plank-specific analysis"""
        issues = []
        suggestions = []
        return issues, suggestions
    
    def analyze_lunge(self, landmarks):
        """Lunge-specific analysis"""
        issues = []
        suggestions = []
        return issues, suggestions
    
    def analyze_deadlift(self, landmarks):
        """Deadlift-specific analysis"""
        issues = []
        suggestions = []
        return issues, suggestions
    
    def detect_exercise(self, image):
        """Detect exercise type using YOLO model"""
        if self.yolo_model is None:
            return 'general'
        
        try:
            # Preprocess image for YOLO
            img = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                pred = self.yolo_model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
            
            # Process detections
            for det in pred:
                if len(det):
                    # Rescale boxes
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    
                    # Return the class with highest confidence
                    max_conf_idx = torch.argmax(det[:, 4])
                    class_id = int(det[max_conf_idx, 5])
                    if class_id < len(self.exercise_classes):
                        return self.exercise_classes[class_id]
            
            return 'general'
        except Exception as e:
            print(f"Exercise detection error: {e}")
            return 'general'
    
    def preprocess_image(self, image):
        """Preprocess image for YOLO input"""
        # Resize and normalize
        img = cv2.resize(image, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img