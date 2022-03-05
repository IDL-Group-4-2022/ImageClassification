from torchvision.io import read_image
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def show_image(imgs_path: str, imgidx: int, df: pd.DataFrame) -> None:
    img_name = df.index[imgidx]
    print(img_name)
    img_path = Path(imgs_path) / f"im{img_name}.jpg"
    image = read_image(str(img_path))
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def test_show_image():
    show_image(
        'resources/data/original/dl2021-image-corpus-proj/images',
        0,
        pd.read_csv(
            'resources/data/generated/train.csv',
            index_col='im_name'
        )
    )
