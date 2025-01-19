import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.data_loader import MarioPartyDataset
import os
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch.nn as nn
import json
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MarioPartyDataset(root_dir='data/train', transform=transform)
print(f"Total images: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")
print(f"Classes: {dataset.classes}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = MarioPartyDataset(root_dir='data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = float('inf')
    
    original_dataset = train_loader.dataset.dataset
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if total % 100 == 0:
                print(f'Batch stats - Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss/len(train_loader)
        accuracy = 100.*correct/total
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'models/best_model.pth')
            
            with open('models/class_mapping.json', 'w') as f:
                json.dump(original_dataset.class_to_idx, f)

def predict_image(image_path, model, class_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    return predicted_class, confidence

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    print("\nModel saved to: models/best_model.pth")
    print("Class mapping saved to: models/class_mapping.json")

def load_model():
    checkpoint = torch.load('models/best_model.pth')
    with open('models/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, class_mapping