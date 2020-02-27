import data
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

unet = model.UNet2D()

optimizer = optim.Adam(unet.parameters(), lr=0.0001)

batch_size = 2

train_set = data.CoralDataset2D(sample_dir="data/train/image", 
                                label_dir="data/train/label",
                                transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

def num_correct(predictions, labels):
    return predictions.argmax(1).eq(labels).sum().item()

for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in tqdm(train_loader):
        samples, labels = batch
        predictions = unet(samples)
        labels = torch.squeeze(labels, 1).long()
        loss = F.cross_entropy(predictions, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * batch_size
        total_correct += num_correct(predictions, labels)

    print(f'epoch: {epoch} total correct: {total_correct}, loss: {total_loss}')