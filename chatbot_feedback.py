import random

class ChatbotFeedback:
    """
    A class to generate natural language feedback for workout form analysis
    """
    
    def __init__(self):
        # Simple feedback templates
        self.feedback_templates = {
            "proper": [
                "Excellent form! Your technique is spot on! 🎉",
                "Perfect execution! You're demonstrating proper form! ✅",
                "Great job! Your form matches proper technique standards! 👏"
            ],
            "good": [
                "Good form overall! Just minor adjustments needed! 👍",
                "Solid technique with room for small refinements! 💪",
                "Well done! Your form is good with some areas to polish! ✨"
            ],
            "needs_work": [
                "Your form needs some improvement. Focus on these areas: 💪",
                "There are several areas that need attention. Let's work on: 📝",
                "Some adjustments needed for better form. Pay attention to: 🔍"
            ],
            "improper": [
                "Significant form issues detected. Important corrections needed: 🔴",
                "Your form needs major adjustments. Critical areas to fix: ⚠️",
                "Several form corrections required. Focus on these key areas: 🛠️"
            ]
        }
    
    def generate_form_feedback(self, analysis_data):
        """Generate friendly feedback based on analysis results"""
        
        status = analysis_data.get('status', 'needs_work')
        issues = analysis_data.get('issues', [])
        suggestions = analysis_data.get('suggestions', [])
        exercise_type = analysis_data.get('exercise_type', 'exercise')
        similarity = analysis_data.get('reference_similarity', 0)
        
        # Convert numpy types to native Python types for JSON serialization
        if hasattr(similarity, 'item'):
            similarity = similarity.item()  # Convert numpy float to native float
        similarity = float(similarity)  # Ensure it's a native float
        
        feedback_messages = []
        
        # Add status-based opening message
        if status in self.feedback_templates:
            opening_message = random.choice(self.feedback_templates[status])
            feedback_messages.append(opening_message)
        else:
            feedback_messages.append("I've analyzed your workout form. Here's my feedback:")
        
        # Add exercise type info
        exercise_name = exercise_type.capitalize() if exercise_type != 'general' else 'Workout'
        feedback_messages.append(f"**Exercise:** {exercise_name}")
        
        # Add reference similarity if available
        if similarity > 0:
            confidence_level = "High" if similarity > 0.7 else "Medium" if similarity > 0.5 else "Low"
            feedback_messages.append(f"**Confidence:** {confidence_level} ({similarity:.1%} match with reference poses)")
        
        # Add specific issues if any
        if issues and len(issues) > 0:
            feedback_messages.append("**Areas for improvement:**")
            for issue in issues[:3]:  # Limit to 3 main issues
                feedback_messages.append(f"• {issue}")
        
        # Add suggestions
        if suggestions and len(suggestions) > 0:
            feedback_messages.append("**Suggestions:**")
            for suggestion in suggestions[:3]:  # Limit to 3 suggestions
                feedback_messages.append(f"• {suggestion}")
        
        # Add encouragement
        if status in ["proper", "good"]:
            feedback_messages.append("Keep up the great work! Your dedication to proper form is paying off! 💪")
        else:
            feedback_messages.append("Don't get discouraged! Everyone improves with practice. Keep working at it! 📈")
        
        print(f"DEBUG: Final feedback messages: {feedback_messages}")
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