import data
import model
import torch.optim as optim
from torch.utils.data import DataLoader

unet = model.UNet3D()

optimizer = optim.Adam(unet.parameters(), lr=0.0001)

batch_size = 2

train_set = data.CoralDataset2D(sample_dir="data/train/image", label_dir="data/train/label")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)