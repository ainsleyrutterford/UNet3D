import data
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

unet = model.UNet2D().to(device)

optimizer = optim.Adam(unet.parameters(), lr=0.0001)

batch_size = 2

train_set = data.CoralDataset2D(sample_dir="data/train/image", 
                                label_dir="data/train/label",
                                transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=cpu_count())

for epoch in range(5):
    total_loss = 0
    for samples, labels in train_loader:
        samples = samples.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, 1).long()
        predictions = unet(samples)
        loss = F.cross_entropy(predictions, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * batch_size

    print(f'epoch: {epoch} loss: {total_loss}')