import os
from torch.utils.data import Dataset
from PIL import Image

class MarioPartyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.file_list = []
        
        for game_dir in os.listdir(root_dir):
            game_path = os.path.join(root_dir, game_dir)
            if os.path.isdir(game_path):
                for image_file in os.listdir(game_path):
                    if image_file.endswith(('.webp', '.jpg', '.png')):
                        minigame_name = os.path.splitext(image_file)[0].split('_')[0]
                        if minigame_name not in self.classes:
                            self.classes.append(minigame_name)
                        self.file_list.append((os.path.join(game_path, image_file), minigame_name))
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, minigame_name = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[minigame_name]
        return image, label
