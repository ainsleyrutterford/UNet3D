import glob
import os
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CoralDataset2D(Dataset):
    """2D Coral slices dataset."""

    def __init__(self, sample_dir, label_dir, transform=None):
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(f"{self.sample_dir}/*.png"))

    def __getitem__(self, idx):
        f = sorted(glob.glob(f"{self.sample_dir}/*.png"))[idx]
        name = os.path.abspath(f)
        sample = io.imread(name)
        sample = transforms.functional.to_pil_image(sample)

        f = sorted(glob.glob(f"{self.label_dir}/*.png"))[idx]
        name = os.path.abspath(f)
        label = io.imread(name)
        threshold = label < 0.5
        label[threshold] = 0
        label = transforms.functional.to_pil_image(label)

        if self.transform:
            sample = self.transform(sample)
            label = self.transform(label)

        return sample, label