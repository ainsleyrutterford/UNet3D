import model
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import io, img_as_ubyte

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

unet = model.UNet2D().to(device)
unet.load_state_dict(torch.load('save.pt', map_location=device))
unet.eval()

t = transforms.ToTensor()
sample = t(io.imread('test.png'))
sample = torch.unsqueeze(sample, 0)
sample = sample.to(device)

with torch.no_grad():
    prediction = unet(sample)
    image = F.softmax(prediction, dim=1)[0,1,:,:]
    image = image.cpu().numpy()
    io.imsave('out.png', img_as_ubyte(image))