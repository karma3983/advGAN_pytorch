import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net

use_cuda=True
image_nc=1
epochs = 1 #本来は60
batch_size = 128 #本来は128
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./MNIST_target_model.pth"
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True) #学習用6万枚
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
advGAN = AdvGAN_Attack(device, #CUDA
                          targeted_model, #MNIST_target_net().to(device)
                          model_num_labels, #10
                          image_nc, #1
                          BOX_MIN, #0
                          BOX_MAX) #1

#dataloader = DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=1), エポック数60
advGAN.train(dataloader, epochs)
