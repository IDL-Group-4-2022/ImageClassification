import torch
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms


def test_images(
    images_path: Path,
    model_path: Path,
    labels: list[str],
    target_csv_path: Path
) -> None:
    """Get predictions for the test set

    Args:
        images_path (Path): image path
        model_path (Path): model path
        labels (list[str]): labels
        target_csv_path (Path): path to write the results
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load(model_path).to(device)
    model.eval()
    matrix = []
    for image_path in Path(images_path).glob('*.jpg'):
        image = Image.open(image_path).convert("RGB")
        tensor = transforms.ToTensor()(image).unsqueeze_(0).to(device)
        predictions = list(np.where(torch.sigmoid(model(tensor)).cpu() > 0.5, 1, 0)[0])
        image_id = image_path.stem[2:]
        matrix.append([image_id] + predictions)
    pd.DataFrame(data=matrix, columns=labels).to_csv(target_csv_path, index=False)


test_images(
    'resources/data/original/image-test-corpus-139ArJI/images',
    'resources/models/Cnn2_final.pytorch',
    pd.read_csv('resources/data/generated/train.csv').columns,
    'resources/data/generated/test.csv'
)
