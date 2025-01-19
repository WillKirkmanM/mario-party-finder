import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import torch.nn.functional as F
import torch.nn as nn

def load_model_and_mapping():
    with open('models/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, len(class_mapping))
    
    checkpoint = torch.load('models/best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, class_mapping

def predict_image(image_path, model, class_mapping):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    idx_to_class = {v: k for k, v in class_mapping.items()}
    predicted_class = idx_to_class[predicted.item()]
    confidence = probabilities[0][predicted].item()
    
    try:
        if '/' in predicted_class:
            game_name, minigame_name = predicted_class.split('/')
        else:
            game_name = "Unknown"
            minigame_name = predicted_class
    except Exception as e:
        print(f"Debug - Full class name: {predicted_class}")
        game_name = "Unknown"
        minigame_name = predicted_class
    
    return game_name, minigame_name, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Mario Party minigame from image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    args = parser.parse_args()

    model, class_mapping = load_model_and_mapping()
    game, minigame, confidence = predict_image(args.image, model, class_mapping)

    if confidence < args.threshold:
        print(f"Warning: Low confidence prediction ({confidence:.2%})")
    
    print(f"Game: {game}")
    print(f"Minigame: {minigame}")
    print(f"Confidence: {confidence:.2%}")