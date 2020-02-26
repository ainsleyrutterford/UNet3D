import model
import torch.optim as optim

unet = model.UNet3D()

optimizer = optim.Adam(unet.parameters(), lr=0.0001)

batch_size = 2