from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.image_dataset import CustomImageDataset
from torch.utils.data import DataLoader


def get_dataloaders(df, test_size, img_dir, batch_size_train, batch_size_test):
    colour_jitter = 0.1
    train_transform = transforms.Compose([
        # transforms.RandomAdjustSharpness(2),
        transforms.ColorJitter(
            brightness=colour_jitter,
            contrast=colour_jitter,
            saturation=colour_jitter,
            hue=colour_jitter,
        ),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 4)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.05),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([transforms.ToTensor()])

    train, dev = train_test_split(df, test_size=test_size)

    train_set = CustomImageDataset(train, img_dir, transform=train_transform)
    dev_set = CustomImageDataset(dev, img_dir, transform=test_transform)

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size_train, shuffle=True
    )
    dev_loader = DataLoader(
        dataset=dev_set, batch_size=batch_size_test, shuffle=True
    )

    return train_loader, dev_loader
