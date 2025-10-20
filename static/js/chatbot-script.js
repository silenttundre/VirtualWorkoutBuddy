document.addEventListener('DOMContentLoaded', function() {
    const chatbotWidget = document.getElementById('chatbot-widget');
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotCard = document.getElementById('chatbot-card');
    const chatbotCardBody = document.getElementById('chatbot-card-body');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotSend = document.getElementById('chatbot-send');
    const closeButton = document.querySelector('.close-chatbot');
    
    // Check if elements exist
    if (!chatbotToggle || !chatbotCard) {
        console.error('Chatbot elements not found');
        return;
    }
    
    chatbotToggle.addEventListener('click', function() {
        console.log('Chatbot toggle clicked');
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
        
        // Send to Ollama RAG backend
        processUserMessageWithRAG(message);
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
    
    async function processUserMessageWithRAG(message) {
        const typingIndicator = showTypingIndicator();
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (chatbotCardBody.contains(typingIndicator)) {
                chatbotCardBody.removeChild(typingIndicator);
            }
            
            if (data.response) {
                addChatMessage(data.response, 'bot-message');
            } else {
                addChatMessage("I'm sorry, I couldn't process your request. Please try again.", 'bot-message');
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            
            if (chatbotCardBody.contains(typingIndicator)) {
                chatbotCardBody.removeChild(typingIndicator);
            }
            
            addChatMessage("I'm having trouble connecting right now. Please try again later.", 'bot-message');
        }
    }
    
    // Auto-open after delay
    setTimeout(() => {
        if (!chatbotCard.classList.contains('active')) {
            chatbotToggle.classList.add('pulse-animation');
            setTimeout(() => {
                chatbotToggle.classList.remove('pulse-animation');
                chatbotCard.classList.add('active');
                addChatMessage("Hello! I'm your Virtual Workout Buddy assistant. I can help you with exercise form analysis, workout questions, and fitness guidance. How can I help you today?", 'bot-message');
            }, 1000);
        }
    }, 5000);
});