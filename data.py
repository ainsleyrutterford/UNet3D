import glob
import os
from torch.utils.data import Dataset, DataLoader

class CoralDataset2D(Dataset):
    """2D Coral slices dataset."""

    def __init__(self, sample_dir, label_dir, transform=None):
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(f"{self.sample_dir}/*.png"))

    def __getitem__(self, idx):
        f = sorted(glob.glob(f"{self.sample_dir}/*.png")[idx])
        name = os.path.basename(f)
        sample = io.imread(name)

        f = sorted(glob.glob(f"{self.label_dir}/*.png")[idx])
        name = os.path.basename(f)
        label = io.imread(name)

        if self.transform:
            sample = self.transform(sample)

        return {'sample': torch.tensor(sample), 'label': torch.tensor(label)}