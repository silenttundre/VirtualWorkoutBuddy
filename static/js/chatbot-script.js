document.addEventListener('DOMContentLoaded', function() {
    const chatbotWidget = document.getElementById('chatbot-widget');
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotCard = document.getElementById('chatbot-card');
    const chatbotCardBody = document.getElementById('chatbot-card-body');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotSend = document.getElementById('chatbot-send');
    const closeButton = document.querySelector('.close-chatbot');
    
    chatbotToggle.addEventListener('click', function() {
        chatbotCard.classList.toggle('active');
        if (chatbotCard.classList.contains('active')) {
            chatbotInput.focus();
        }
    });
    
    closeButton.addEventListener('click', function() {
        chatbotCard.classList.remove('active');
    });
    
    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (message === '') return;
        
        addChatMessage(message, 'user-message');
        chatbotInput.value = '';
        processUserMessage(message);
    }
    
    chatbotSend.addEventListener('click', sendMessage);
    chatbotInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    function addChatMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        const messagePara = document.createElement('p');
        messagePara.textContent = text;
        
        messageDiv.appendChild(messagePara);
        chatbotCardBody.appendChild(messageDiv);
        chatbotCardBody.scrollTop = chatbotCardBody.scrollHeight;
    }
    
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chatbot-typing';
        typingDiv.innerHTML = `
            <div class="chatbot-typing-dot"></div>
            <div class="chatbot-typing-dot"></div>
            <div class="chatbot-typing-dot"></div>
        `;
        chatbotCardBody.appendChild(typingDiv);
        chatbotCardBody.scrollTop = chatbotCardBody.scrollHeight;
        return typingDiv;
    }
    
    function processUserMessage(message) {
        const typingIndicator = showTypingIndicator();
        
        setTimeout(() => {
            if (chatbotCardBody.contains(typingIndicator)) {
                chatbotCardBody.removeChild(typingIndicator);
            }
            
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('upload') || lowerMessage.includes('image') || lowerMessage.includes('photo') || lowerMessage.includes('picture')) {
                addChatMessage("Great! You can upload an image by clicking the 'Try It Now' button on the page, or you can drag and drop an image directly into the upload area.", 'bot-message');
                
                setTimeout(() => {
                    addChatMessage("Make sure to select the correct exercise type from the dropdown menu before uploading.", 'bot-message');
                }, 1000);
                
                const analyzerSection = document.getElementById('analyzer');
                if (analyzerSection) {
                    setTimeout(() => {
                        analyzerSection.scrollIntoView({ behavior: 'smooth' });
                    }, 1500);
                }
            } 
            else if (lowerMessage.includes('exercise') || lowerMessage.includes('workout')) {
                addChatMessage("I can analyze several types of exercises including squats, push-ups, planks, lunges, and deadlifts. Which one are you interested in?", 'bot-message');
            }
            else if (lowerMessage.includes('squat')) {
                addChatMessage("For squats, I'll check your knee alignment, back posture, and depth. Make sure your knees track over your toes and don't extend past them. Your back should remain straight, not rounded.", 'bot-message');
            }
            else if (lowerMessage.includes('pushup') || lowerMessage.includes('push-up') || lowerMessage.includes('push up')) {
                addChatMessage("For push-ups, I'll analyze your arm position, back alignment, and neck posture. Your elbows should be at about a 45-degree angle to your body, your back straight, and your neck neutral.", 'bot-message');
            }
            else if (lowerMessage.includes('plank')) {
                addChatMessage("For planks, I'll check your shoulder alignment, hip position, and overall body line. Your shoulders should be directly over your elbows, hips neither too high nor too low, and your body should form a straight line.", 'bot-message');
            }
            else if (lowerMessage.includes('help')) {
                addChatMessage("I can help you with workout form analysis, exercise recommendations, and answer questions about the application. Feel free to ask me anything!", 'bot-message');
            }
            else if (lowerMessage.includes('thank')) {
                addChatMessage("You're welcome! I'm happy to help. Let me know if you need anything else.", 'bot-message');
            }
            else if (lowerMessage.includes('hi') || lowerMessage.includes('hello')) {
                addChatMessage("Hello! How can I help you with your workout analysis today?", 'bot-message');
            }
            else {
                addChatMessage("I'm here to help with your workout form analysis. You can ask me about specific exercises, how to upload images, or any other questions you have about using this application.", 'bot-message');
            }
        }, 1000);
    }
    
    setTimeout(() => {
        if (!chatbotCard.classList.contains('active')) {
            chatbotToggle.classList.add('pulse-animation');
            setTimeout(() => {
                chatbotToggle.classList.remove('pulse-animation');
                chatbotCard.classList.add('active');
                addChatMessage("Would you like to analyze your workout form? I can help you upload an image and provide feedback!", 'bot-message');
            }, 1000);
        }
    }, 5000);
});