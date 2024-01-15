import os

import numpy as np

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio

import matplotlib.pyplot as plt

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
def tensor2PILimage(output_img):
    unloader = transforms.ToPILImage()
    output_img = output_img.cpu().clone()
    output_img = output_img.squeeze(0)
    output_img = unloader(output_img)
    output_img = np.array(output_img)
    return output_img

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载已保存的模型
model = DnCNN()
model.load_state_dict(torch.load('dncnn_model_6.pth'))
model.eval()

# 加载测试集
running_psnr = 0.0
running_psnr_test = 0.0
test_noisy_dir = "C:\\Users\\Administrator\\Desktop\\视觉检测作业\\dataset\\test\\NOISE"
test_clean_dir = "C:\\Users\\Administrator\\Desktop\\视觉检测作业\\dataset\\test\\GT"
noisy_images = os.listdir(test_noisy_dir)
clean_images = os.listdir(test_clean_dir)
for i in range(len(noisy_images)):
    noisy_path = os.path.join(test_noisy_dir,noisy_images[i])
    clean_path = os.path.join(test_clean_dir, clean_images[i])
    noisy_img = Image.open(noisy_path)
    clean_img = Image.open(clean_path)
    noisy_img = transform(noisy_img).unsqueeze(0)
    clean_img = transform(clean_img).unsqueeze(0)

    with torch.no_grad():
        v_img = model(noisy_img)
        output_img = noisy_img - v_img
        output_img = tensor2PILimage(output_img)
        clean_img = tensor2PILimage(clean_img)
        noisy_img = tensor2PILimage(noisy_img)
        psnr =peak_signal_noise_ratio(clean_img,output_img)
        psnr_test = peak_signal_noise_ratio(clean_img,noisy_img)
        running_psnr += psnr
        running_psnr_test += psnr_test
print(f"未经处理的峰值信噪比:{running_psnr_test/len(noisy_images)}")
print(f"峰值信噪比:{running_psnr/len(noisy_images)}")
# test_image = Image.open(test_image_path)
# test_image = transform(test_image).unsqueeze(0)  # 添加 batch 维度

# # 在模型上进行推断
# with torch.no_grad():
#     denoised_image = model(test_image)
#
# # 如果使用GPU，将结果转移到CPU并转为NumPy数组
# denoised_image = denoised_image.cpu().numpy()
#
# # 这里可以添加显示或保存结果的代码
#
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(test_image.squeeze().numpy(), cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(denoised_image.squeeze(), cmap='gray')
# plt.title('Denoised Image')
#
# plt.show()