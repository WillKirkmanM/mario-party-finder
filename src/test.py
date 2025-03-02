import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import torch.nn.functional as F
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
class_mapping = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_and_mapping():
    with open('models/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_mapping))
    
    checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_mapping

def predict_image(image_path, model, class_mapping):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = Image.open(image_path).convert('RGB')
    image_for_display = image.copy()
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        
        top_probs, top_idx = torch.topk(probabilities, 3, dim=1)
        
    results = []
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    for i in range(3):
        predicted_class = idx_to_class[top_idx[0][i].item()]
        confidence = top_probs[0][i].item()
        
        try:
            if '/' in predicted_class:
                game_name, minigame_name = predicted_class.split('/')
            else:
                game_name = "Unknown"
                minigame_name = predicted_class
        except Exception:
            game_name = "Unknown"
            minigame_name = predicted_class
        
        results.append({
            'game': game_name,
            'minigame': minigame_name,
            'confidence': f"{confidence:.2%}"
        })
    
    buffered = BytesIO()
    image_for_display.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return results, img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results, img_str = predict_image(filepath, model, class_mapping)
            return render_template(
                'result.html', 
                results=results, 
                image_b64=img_str
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    model, class_mapping = load_model_and_mapping()
    print("Model loaded successfully! Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)