import random

class ChatbotFeedback:
    """
    A class to generate natural language feedback for workout form analysis
    """
    
    def __init__(self):
        # Greeting templates
        self.greetings = [
            "Hi there! I've analyzed your {exercise} form.",
            "Thanks for sharing your {exercise} position! Here's my analysis.",
            "I've checked your {exercise} form. Let me share what I found.",
            "Great effort with your {exercise}! Here's my feedback on your form."
        ]
        
        # Proper form templates
        self.proper_form_messages = [
            "Your {exercise} form looks excellent! Great job maintaining proper technique.",
            "Perfect {exercise} form! You're demonstrating excellent technique.",
            "Impressive {exercise} form! Your alignment and positioning are spot on.",
            "Excellent work! Your {exercise} form shows great attention to detail."
        ]
        
        # Improper form templates
        self.improper_form_intros = [
            "I noticed a few things that could help improve your {exercise} form:",
            "Your {exercise} form could use some adjustments. Here's what I observed:",
            "I see some opportunities to enhance your {exercise} technique:",
            "Let me help you refine your {exercise} form with these observations:"
        ]
        
        # Issue templates
        self.issue_templates = {
            "shoulders": [
                "Your shoulders aren't quite level, which can create imbalance.",
                "I noticed your shoulders aren't aligned properly.",
                "Your shoulder positioning needs some adjustment for better form."
            ],
            "knees": [
                "Your knee alignment could be improved for better stability.",
                "The angle of your knees isn't optimal for this exercise.",
                "Your knees need some adjustment to prevent potential strain."
            ],
            "back": [
                "Your back position could be adjusted for better form.",
                "I noticed your back isn't maintaining the ideal alignment.",
                "Your back positioning needs some refinement for proper technique."
            ],
            "hips": [
                "Your hip position could be improved for better form.",
                "Your hips aren't quite in the optimal position for this exercise.",
                "The alignment of your hips needs some adjustment."
            ],
            "alignment": [
                "Your body alignment needs some adjustment for proper form.",
                "Your overall alignment could be improved for better results.",
                "I noticed some alignment issues that could be refined."
            ]
        }
        
        # Suggestion templates
        self.suggestion_templates = {
            "shoulders": [
                "Try focusing on keeping your shoulders level and aligned.",
                "Work on maintaining even height with both shoulders.",
                "Practice with a mirror to check your shoulder alignment."
            ],
            "knees": [
                "Focus on proper knee alignment over your ankles.",
                "Try to maintain the recommended angle in your knees.",
                "Be mindful of your knee position throughout the movement."
            ],
            "back": [
                "Concentrate on maintaining a neutral spine position.",
                "Focus on keeping your back straight throughout the exercise.",
                "Try engaging your core to support proper back alignment."
            ],
            "hips": [
                "Work on positioning your hips at the proper height.",
                "Focus on hip alignment to maximize exercise benefits.",
                "Pay attention to your hip position throughout the movement."
            ],
            "alignment": [
                "Practice with a mirror to check your overall alignment.",
                "Focus on maintaining proper alignment from head to toe.",
                "Try to be mindful of your body's position throughout the exercise."
            ]
        }
        
        # Encouragement templates
        self.encouragement_proper = [
            "Keep up the great work! Consistent proper form helps prevent injuries and maximizes results.",
            "You're doing fantastic! Maintaining this form will help you get the most out of your workouts.",
            "Excellent job! Your attention to proper form will lead to better long-term results.",
            "Well done! Your commitment to proper technique will help you progress safely and effectively."
        ]
        
        self.encouragement_improper = [
            "Don't worry! Form takes practice. These adjustments will help you improve quickly.",
            "Keep practicing! With these adjustments, you'll see improvements in your form soon.",
            "Everyone starts somewhere! These small changes will make a big difference in your results.",
            "You're on the right track! These refinements will help you get more from your workouts."
        ]
        
        # Exercise-specific templates
        self.exercise_specific_tips = {
            "squat": [
                "Remember to keep your chest up and back straight during squats.",
                "Try to push your knees outward slightly to maintain proper alignment.",
                "Focus on driving through your heels as you stand up from a squat."
            ],
            "pushup": [
                "Keep your core engaged throughout the pushup to maintain proper alignment.",
                "Focus on a controlled descent and ascent for maximum benefit.",
                "Position your hands slightly wider than shoulder-width for standard pushups."
            ],
            "plank": [
                "Focus on creating a straight line from head to heels in your plank.",
                "Engage your core by drawing your navel toward your spine.",
                "Distribute your weight evenly between your forearms and toes."
            ],
            "lunge": [
                "Keep your front knee aligned with your ankle, not extending past your toes.",
                "Maintain an upright torso throughout the lunge movement.",
                "Step far enough forward to create proper knee angles."
            ],
            "deadlift": [
                "Initiate the movement by hinging at the hips, not by bending the knees first.",
                "Keep the bar close to your body throughout the movement.",
                "Maintain a neutral spine position from start to finish."
            ],
            "general": [
                "Focus on controlled movements rather than speed for better form.",
                "Proper breathing helps maintain form - exhale during exertion.",
                "Regular practice with proper form will lead to better results."
            ]
        }
        
        # Closing templates
        self.closing_messages = [
            "Keep up the good work! I'm here if you need more form checks.",
            "Looking forward to seeing your progress! Upload another image anytime.",
            "Remember, consistent practice with proper form leads to the best results!",
            "Feel free to upload more workout images for feedback anytime."
        ]
    
    def _get_issue_category(self, issue_text):
        """Determine the category of an issue based on keywords"""
        issue_text = issue_text.lower()
        
        if any(keyword in issue_text for keyword in ["shoulder", "shoulders"]):
            return "shoulders"
        elif any(keyword in issue_text for keyword in ["knee", "knees"]):
            return "knees"
        elif any(keyword in issue_text for keyword in ["back", "spine"]):
            return "back"
        elif any(keyword in issue_text for keyword in ["hip", "hips"]):
            return "hips"
        else:
            return "alignment"
    
    def generate_greeting(self, exercise_type):
        """Generate a greeting message based on exercise type"""
        exercise_name = exercise_type.capitalize() if exercise_type != "general" else "workout"
        return random.choice(self.greetings).format(exercise=exercise_name)
    
    def generate_form_feedback(self, analysis_result):
        """Generate detailed feedback based on analysis results"""
        exercise_type = analysis_result.get("exercise_type", "general")
        exercise_name = exercise_type.capitalize() if exercise_type != "general" else "workout"
        status = analysis_result.get("status", "")
        issues = analysis_result.get("issues", [])
        suggestions = analysis_result.get("suggestions", [])
        
        feedback_messages = []
        
        # Add greeting
        feedback_messages.append(self.generate_greeting(exercise_type))
        
        # Add form assessment
        if status == "proper":
            feedback_messages.append(random.choice(self.proper_form_messages).format(exercise=exercise_name))
        else:
            feedback_messages.append(random.choice(self.improper_form_intros).format(exercise=exercise_name))
            
            # Add issues and suggestions with natural language variations
            for i, issue in enumerate(issues):
                category = self._get_issue_category(issue)
                issue_template = random.choice(self.issue_templates.get(category, self.issue_templates["alignment"]))
                feedback_messages.append(f"• {issue_template}")
                
                if i < len(suggestions):
                    suggestion_template = random.choice(self.suggestion_templates.get(category, self.suggestion_templates["alignment"]))
                    feedback_messages.append(f"  → {suggestion_template}")
        
        # Add exercise-specific tip
        if exercise_type in self.exercise_specific_tips:
            feedback_messages.append("\nHelpful tip: " + random.choice(self.exercise_specific_tips[exercise_type]))
        
        # Add encouragement
        if status == "proper":
            feedback_messages.append("\n" + random.choice(self.encouragement_proper))
        else:
            feedback_messages.append("\n" + random.choice(self.encouragement_improper))
        
        # Add closing message
        feedback_messages.append("\n" + random.choice(self.closing_messages))
        
        return feedback_messages
    
    def generate_error_message(self, error):
        """Generate a helpful message when an error occurs"""
        if "No pose detected" in error:
            return [
                "I couldn't detect a clear pose in your image.",
                "This might be due to lighting, clothing, or camera angle.",
                "Try uploading another image with better lighting and a clear view of your full body.",
                "Make sure your whole body is visible in the frame for the best analysis."
            ]
        elif "Could not read image" in error:
            return [
                "I had trouble processing your image.",
                "Please make sure you're uploading a valid image file (JPEG, PNG, etc.).",
                "Try uploading a different image with a common format."
            ]
        else:
            return [
                "Something went wrong while analyzing your image.",
                "Please try uploading a different image with a clear view of your exercise position.",
                "Make sure your whole body is visible in the frame for the best analysis."
            ]