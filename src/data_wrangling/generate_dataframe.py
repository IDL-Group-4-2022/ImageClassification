from pathlib import Path
import pandas as pd

LABELS_DIR = 'resources/data/original/dl2021-image-corpus-proj/annotations/'

label_files = list(Path(LABELS_DIR).glob('*'))
label_names = [f.stem for f in label_files]

img_labels = {}

for label, label_file in zip(label_names, label_files):
    with open(label_file) as f:
        ids = f.readlines()
    for i in ids:
        i = int(i)
        if i not in img_labels:
            img_labels[i] = []
        img_labels[i].append(label)

df = pd.DataFrame([img_labels], index=['labels']).T
df = (
    df
    .drop(['labels'], axis=1)
    .join(df.labels.str.join('|').str.get_dummies())
)

df.to_csv('resources/data/generated/train.csv')
