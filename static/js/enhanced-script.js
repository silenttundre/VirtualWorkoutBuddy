document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileUpload = document.getElementById('file-upload');
    
    if (uploadArea && fileUpload) {
        uploadArea.style.cursor = 'pointer';
        uploadArea.addEventListener('click', function() {
            fileUpload.click();
        });

        fileUpload.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                handleFile(this.files[0]);
            }
        });
    }
});

function handleFile(file) {
    const loadingElement = document.getElementById('loading');
    const uploadContainer = document.getElementById('upload-container');
    
    if (uploadContainer) uploadContainer.style.display = 'none';
    if (loadingElement) loadingElement.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('exercise_type', 'auto');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (loadingElement) loadingElement.style.display = 'none';
        
        if (data.error) {
            alert('Error: ' + data.error);
            if (uploadContainer) uploadContainer.style.display = 'flex';
            return;
        }

        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }

        const resultImage = document.getElementById('result-image');
        if (resultImage && data.image) {
            resultImage.src = `data:image/jpeg;base64,${data.image}`;
            resultImage.style.display = 'block';
        }

        const chatbotMessages = document.getElementById('chatbot-messages');
        if (chatbotMessages) {
            chatbotMessages.innerHTML = '';
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            messageDiv.innerHTML = `<p>${data.analysis.assessment}</p>`;
            chatbotMessages.appendChild(messageDiv);
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        if (loadingElement) loadingElement.style.display = 'none';
        if (uploadContainer) uploadContainer.style.display = 'flex';
        alert('Upload failed: ' + error);
    });
}

