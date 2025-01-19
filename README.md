<p align="center">
  <img src="https://mario.wiki.gallery/images/thumb/8/8a/Super_Mario_Party_Logo.png/1200px-Super_Mario_Party_Logo.png" width=200 />
</p>
<h1 align="center">Mario Party Finder</h1>

<p align="center">A Convolutional Neural Network to Predict Any Mario Party Minigame</p>

<p align="center">
  <img src="https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/10/Mario-Party-Minigames.jpg" />
</p>

<p align="center">
This project is a neural network application designed to classify images based on the game and minigame from Mario Party. The model is built using PyTorch and includes various components for data loading, preprocessing, and training.
</p>

## Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
python src/train.py
```

### Inference
```bash
python src/test.py --image path/to/image.jpg
```

### Data Collection
```bash
python src/scraping/scrape.py
```

## Model Architecture
| Component | Specification |
|-----------|---------------|
| Backbone | ResNet18 (pretrained on ImageNet) |
| Architecture | 18-layer CNN with residual connections |
| Input Size | 224x224x3 RGB images |
| Output | Multi-class classification (softmax) |
| Optimizer | Adam (lr=1e-4) |
| Loss Function | CrossEntropy |
| Batch Size | 32 |
| Training Duration | 10 epochs |
| Learning Schedule | ReduceLROnPlateau (patience=2) |
| Data Augmentation | Resize, Normalize (ImageNet stats) |

## Dataset
The dataset includes screenshots from various Mario Party games:

Mario Party 1-10
Mario Party DS
Mario Party Advance
Mario Party Island Tour
Mario Party Star Rush
Super Mario Party
Mario Party Superstars

## Requirements
torch
torchvision
Pillow
numpy
requests
beautifulsoup4

## Dependencies
```
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install tqdm

# Run main script
python src/main.py
```

## Project Structure

```
mario-party-classifier
├── src
│   ├── models
│   │   ├── __init__.py
│   │   └── neural_net.py
│   ├── data
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   └── main.py
├── tests
│   ├── __init__.py
│   └── test_neural_net.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/WillKirkmanM/mario-party-finder
   cd mario-party-finder
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/train.py
   ```

## Details

- The `DataLoader` class in `src/data/data_loader.py` handles loading and preprocessing the image data.
- The `NeuralNetwork` class in `src/models/neural_net.py` is responsible for building, training, and evaluating the model.
- Use the `preprocessing` functions in `src/utils/preprocessing.py` for any image preprocessing needs.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.