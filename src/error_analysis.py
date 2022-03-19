import torch
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


def predict_image(
    img_path: Path, model_path: Path, labels: list[str]
) -> None:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load(model_path).to(device)
    model.eval()
    image = Image.open(img_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze_(0).to(device)
    predictions = np.where(torch.sigmoid(model(tensor)).cpu() > 0.5, 1, 0)[0]
    print([labels[i] for i, _ in enumerate(predictions) if i == 1])
    plt.imshow(image)
    plt.show()


def test_predict_image():
    predict_image(
        'resources/data/original/dl2021-image-corpus-proj/images/im311.jpg',
        'resources/models/Transferred.pytorch',
        pd.read_csv('resources/data/generated/train.csv').columns,
    )


test_predict_image()
