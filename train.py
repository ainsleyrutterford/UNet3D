import data
import model
import torch.optim as optim

unet = model.UNet3D()

optimizer = optim.Adam(unet.parameters(), lr=0.0001)

batch_size = 2

train_set = data.CoralDataset2D(sample_dir="data/image", label_dir="data/label")