async function predictSpam() {
    const messageInput = document.getElementById('message-input');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const spinner = predictBtn.querySelector('.loading-spinner');
    
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message to analyze');
        return;
    }
    
    predictBtn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Failed to connect to server. Make sure the backend is running.');
    } finally {
        predictBtn.disabled = false;
        btnText.style.display = 'inline-block';
        spinner.style.display = 'none';
    }
}

function displayResult(data) {
    document.getElementById('message-preview').textContent = data.message;
    
    const badge = document.getElementById('prediction-badge');
    const badgeClass = data.is_spam ? 'spam' : 'not-spam';
    const badgeText = data.is_spam ? '🚫 SPAM DETECTED' : '✅ LEGITIMATE MESSAGE';
    
    badge.textContent = badgeText;
    badge.className = `prediction-badge ${badgeClass}`;
    
    document.getElementById('confidence-value').textContent = data.confidence + '%';
    document.getElementById('meter-fill').style.width = data.confidence + '%';
    
    document.querySelector('.input-section').style.display = 'none';
    document.getElementById('result-section').style.display = 'block';
}

function resetForm() {
    document.querySelector('.input-section').style.display = 'block';
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('message-input').value = '';
}