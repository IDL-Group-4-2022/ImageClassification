import torch
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


def predict_image(
    images_path: Path, image_id: int, model_path: Path, df: pd.DataFrame
) -> None:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load(model_path).to(device)
    model.eval()
    labels = df.columns
    image = Image.open(f"{images_path}im{image_id}.jpg").convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze_(0).to(device)
    predictions = np.where(torch.sigmoid(model(tensor)).cpu() > 0.5, 1, 0)[0]
    actual = df.loc[image_id]
    print(
        "actual: "
        f"{[labels[i] for i, onehot in enumerate(actual) if onehot == 1]}"
    )
    print(
        "Predicted: "
        f"{[labels[i] for i, onehot in enumerate(predictions) if onehot == 1]}"
    )
    plt.imshow(image)
    plt.show()


def test_predict_image():
    predict_image(
        'resources/data/original/dl2021-image-corpus-proj/images/',
        200,
        'resources/models/Cnn2_final.pytorch',
        pd.read_csv('resources/data/generated/train.csv',index_col='im_name'),
    )


test_predict_image()
