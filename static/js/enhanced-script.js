// enhanced-script.js
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements from base script
    const uploadArea = document.getElementById('upload-area');
    const fileUpload = document.getElementById('file-upload');
    const uploadContainer = document.getElementById('upload-container');
    const loadingElement = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const chatbotMessages = document.getElementById('chatbot-messages');
    //const tryAgainBtn = document.getElementById('try-again-btn');
    
    // Exercise selection functionality
    let selectedExercise = 'general';
    
    // Add exercise selector to upload area
    const exerciseSelector = document.createElement('div');
    exerciseSelector.className = 'exercise-selector';
    exerciseSelector.innerHTML = `
        <label for="exercise-select">Select exercise type:</label>
        <select id="exercise-select">
            <option value="general">General Posture</option>
            <option value="squat">Squat</option>
            <option value="pushup">Push-up</option>
            <option value="plank">Plank</option>
            <option value="lunge">Lunge</option>
            <option value="deadlift">Deadlift</option>
        </select>
    `;
    uploadArea.parentNode.insertBefore(exerciseSelector, uploadArea);
    
    // Add upload progress bar
    const progressContainer = document.createElement('div');
    progressContainer.className = 'upload-progress';
    progressContainer.innerHTML = '<div class="upload-progress-bar"></div>';
    uploadArea.appendChild(progressContainer);
    
    // Get new DOM elements
    const exerciseSelect = document.getElementById('exercise-select');
    const progressBar = document.querySelector('.upload-progress-bar');
    
    // Add typing indicator for chatbot
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'chatbot-typing';
    typingIndicator.innerHTML = `
        <div class="chatbot-typing-dot"></div>
        <div class="chatbot-typing-dot"></div>
        <div class="chatbot-typing-dot"></div>
    `;
    
    // Add history section
    const historySection = document.createElement('div');
    historySection.className = 'history-section';
    historySection.innerHTML = `
        <h3 class="history-title">Previous Analyses</h3>
        <div class="history-cards" id="history-cards"></div>
    `;
    document.querySelector('main').appendChild(historySection);
    
    // Event Listeners
    exerciseSelect.addEventListener('change', function() {
        selectedExercise = this.value;
    });
    
        // Add drag and drop event listeners
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Add file input change listener
    fileUpload.addEventListener('change', function() {
        if (this.files.length) {
            handleFile(this.files[0]);
        }
    });

    // Initialize chatbot with first message
    addMessage('Hi there! Upload a workout image and I\'ll analyze your form.', 'bot-message');

    // Enhanced file handling
    function handleFile(file) {
        // Clear previous messages
        chatbotMessages.innerHTML = 'Hi there! Upload a workout image and I\'ll analyze your form.';
        
        // Check if file is an image
        if (!file.type.match('image.*')) {
            addMessage('Please upload an image file (JPEG, PNG, etc.)', 'bot-message');
            return;
        }
    
        // Show loading state
        uploadContainer.style.display = 'none';
        loadingElement.style.display = 'block';
        
        // Show typing indicator
        chatbotMessages.appendChild(typingIndicator);
    
        // Rest of the existing handleFile function remains the same...
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('exercise_type', 'auto'); // Let YOLO detect the exercise
    
        // Send to server with progress tracking
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);
        
        // Progress tracking
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                progressContainer.style.display = 'block';
                progressBar.style.width = percentComplete + '%';
            }
        };
        
        xhr.onload = function() {
            // Hide loading and progress
            loadingElement.style.display = 'none';
            progressContainer.style.display = 'none';
            progressBar.style.width = '0%';
            
            // Remove typing indicator
            if (chatbotMessages.contains(typingIndicator)) {
                chatbotMessages.removeChild(typingIndicator);
            }
            
            // Show results
            resultsSection.style.display = 'block';
            
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                
                if (data.error) {
                    addMessage(`Error: ${data.error}`, 'bot-message');
                    return;
                }

                // Display the annotated image
                if (data.image) {
                    resultImage.src = `data:image/jpeg;base64,${data.image}`;
                }

                // Display analysis results with chatbot feedback
                const analysis = data.analysis;
                const feedback = data.feedback;
                
                // Display feedback messages with typing effect
                if (feedback && feedback.length > 0) {
                    // First message with appropriate styling based on status
                    let messageClass = 'bot-message';
                    if (analysis.status === 'proper') {
                        messageClass = 'proper-message';
                    } else if (analysis.status === 'improper') {
                        messageClass = 'improper-message';
                    }
                    
                    // Display first message immediately
                    simulateTyping(feedback[0], messageClass);
                    
                    // Display remaining messages with delays
                    let delay = 1500;
                    for (let i = 1; i < feedback.length; i++) {
                        if (feedback[i].trim() !== '') {
                            ((index, d) => {
                                setTimeout(() => {
                                    addMessage(feedback[index], 'bot-message');
                                }, d);
                            })(i, delay);
                            delay += 1000 + (feedback[i].length * 10);
                        }
                    }
                } else {
                    // Fallback to basic message if no feedback is provided
                    simulateTyping(analysis.assessment, analysis.status === 'proper' ? 'proper-message' : 'improper-message');
                    
                    // Add issues if any
                    if (analysis.issues && analysis.issues.length > 0) {
                        setTimeout(() => {
                            let issuesMessage = 'Issues detected:';
                            analysis.issues.forEach(issue => {
                                issuesMessage += `\n• ${issue}`;
                            });
                            addMessage(issuesMessage, 'bot-message');
                        }, 1000);
                    }
                    
                    // Add suggestions if any
                    if (analysis.suggestions && analysis.suggestions.length > 0) {
                        setTimeout(() => {
                            let suggestionsMessage = 'Suggestions for improvement:';
                            analysis.suggestions.forEach(suggestion => {
                                suggestionsMessage += `\n• ${suggestion}`;
                            });
                            addMessage(suggestionsMessage, 'bot-message');
                        }, 2000);
                    }
                }
                
                // Add to history
                addToHistory(file, analysis);
            } else {
                console.error('Error:', xhr.statusText);
                addMessage('Sorry, there was an error processing your image. Please try again.', 'bot-message');
            }
        };
        
        xhr.onerror = function() {
            console.error('Request error');
            loadingElement.style.display = 'none';
            progressContainer.style.display = 'none';
            addMessage('Sorry, there was an error processing your image. Please try again.', 'bot-message');
            
            // Remove typing indicator
            if (chatbotMessages.contains(typingIndicator)) {
                chatbotMessages.removeChild(typingIndicator);
            }
        };
        
        xhr.send(formData);
    }
    
    // Simulate typing effect
    function simulateTyping(text, className) {
        // Add typing indicator
        chatbotMessages.appendChild(typingIndicator);
        
        // Create message container but don't add text yet
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        const messagePara = document.createElement('p');
        messageDiv.appendChild(messagePara);
        chatbotMessages.appendChild(messageDiv);
        
        // Remove typing indicator after delay
        setTimeout(() => {
            if (chatbotMessages.contains(typingIndicator)) {
                chatbotMessages.removeChild(typingIndicator);
            }
            
            // Add text
            messagePara.textContent = text;
            
            // Scroll to bottom of messages
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        }, 1500);
    }
    
    // Add to history
    function addToHistory(file, analysis) {
        const historyCards = document.getElementById('history-cards');
        
        // Create a card for this analysis
        const card = document.createElement('div');
        card.className = 'history-card';
        
        // Create a reader to get image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            const timestamp = new Date().toLocaleString();
            
            card.innerHTML = `
                <img src="${e.target.result}" class="history-card-image" alt="Workout image">
                <div class="history-card-content">
                    <h4 class="history-card-title">${selectedExercise.charAt(0).toUpperCase() + selectedExercise.slice(1)}</h4>
                    <div class="history-card-date">${timestamp}</div>
                    <div class="history-card-status ${analysis.status}">${analysis.status.charAt(0).toUpperCase() + analysis.status.slice(1)}</div>
                </div>
            `;
            
            // Add click event to show this analysis again
            card.addEventListener('click', function() {
                // Future enhancement: Show detailed view of this analysis
                addMessage(`You clicked on a previous ${selectedExercise} analysis from ${timestamp}.`, 'bot-message');
            });
            
            // Add to history (at the beginning)
            if (historyCards.firstChild) {
                historyCards.insertBefore(card, historyCards.firstChild);
            } else {
                historyCards.appendChild(card);
            }
        };
        
        reader.readAsDataURL(file);
    }
    
    // Override the original addMessage function
    window.addMessage = function(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        const messagePara = document.createElement('p');
        messagePara.textContent = text;
        
        messageDiv.appendChild(messagePara);
        chatbotMessages.appendChild(messageDiv);
        
        // Scroll to bottom of messages
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    };
    
    // Override the original resetUI function
    window.resetUI = function() {
        // Hide results
        resultsSection.style.display = 'none';
        
        // Clear file input
        fileUpload.value = '';
        
        // Show upload container
        uploadContainer.style.display = 'flex';
        
        // Clear result image
        resultImage.src = '';
        
        // Reset progress bar
        progressContainer.style.display = 'none';
        progressBar.style.width = '0%';
        
        // Reset chatbot messages to initial state
        chatbotMessages.innerHTML = '';
        addMessage('Hi there! Upload a workout image and I\'ll analyze your form.', 'bot-message');
    };

    
});