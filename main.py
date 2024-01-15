import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定义数据加载类
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

        self.noisy_images = os.listdir(noisy_dir)
        self.clean_images = os.listdir(clean_dir)

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_img_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_img_path = os.path.join(self.clean_dir, self.clean_images[idx])

        noisy_img = Image.open(noisy_img_path)
        clean_img = Image.open(clean_img_path)

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img

# 定义DnCNN模型
class DnCNN(nn.Module):
    def __init__(self, num_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)

# Totensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# 路径
noisy_data_path = "C:\\Users\\Administrator\\Desktop\\视觉检测作业\\dataset\\train\\NOISE"
clean_data_path = "C:\\Users\\Administrator\\Desktop\\视觉检测作业\\dataset\\train\\GT"

dataset = DenoisingDataset(noisy_data_path, clean_data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 80#15轮左右较好

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_loss_test = 0.0

    for noisy_img, clean_img in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)

        optimizer.zero_grad()

        outputs = model(noisy_img)
        loss = criterion(outputs, noisy_img-clean_img)
        loss_test = criterion(clean_img,noisy_img)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_test += loss_test.item()

    print(f"未训练的loss:{running_loss_test/ len(dataloader)}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    #画图检测训练效果
    if epoch > 70 or epoch==10 or epoch==20 or epoch==30 or epoch==40 or epoch==50 or epoch==60:
        unloader = transforms.ToPILImage()
        noisy_img_show = noisy_img.cpu().clone()  # clone the tensor
        noisy_img_show = noisy_img_show.squeeze(0)  # remove the fake batch dimension
        noisy_img_show = unloader(noisy_img_show)
        clean_img_show = clean_img.cpu().clone()  # clone the tensor
        clean_img_show = clean_img_show.squeeze(0)  # remove the fake batch dimension
        clean_img_show = unloader(clean_img_show)
        outputs_img = model(noisy_img)
        outputs_img_show = outputs_img.cpu().clone()  # clone the tensor
        outputs_img_show = outputs_img_show.squeeze(0)  # remove the fake batch dimension
        outputs_img_show = unloader(outputs_img_show)
        final_img = noisy_img-outputs_img
        final_img_show = final_img.cpu().clone()  # clone the tensor
        final_img_show = final_img_show.squeeze(0)  # remove the fake batch dimension
        final_img_show = unloader(final_img_show)
        plt.subplot(2, 2, 1)
        plt.imshow(noisy_img_show)
        plt.subplot(2, 2, 2)
        plt.imshow(outputs_img_show)
        plt.subplot(2, 2, 3)
        plt.imshow(clean_img_show)
        plt.subplot(2, 2, 4)
        plt.imshow(final_img_show)
        plt.show()
# 保存模型
torch.save(model.state_dict(), 'dncnn_model_5_2.pth')