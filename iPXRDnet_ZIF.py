#run
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from torch.utils.data import DataLoader, Dataset,TensorDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import pandas as pd
import numpy as np
import torch,math,random
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from time import time, strftime, localtime
from torch.nn.functional import   relu
import csv,json,copy,torch,re


class Inception(nn.Module):
    # c1--c4 is the number of output channels per path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Line 1, single 1x1 convolution layer
        self.p1_1 = nn.Conv1d(in_channels, c1, kernel_size=1)
        # Line 2, 1x1 convolutional layer followed by 3x3 convolutional layer
        self.p2_1 = nn.Conv1d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv1d(c2[0], c2[1], kernel_size=5, padding=2)
        # Line 3, 1x1 convolutional layer followed by 5x5 convolutional layer
        self.p3_1 = nn.Conv1d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv1d(c3[0], c3[1], kernel_size=23, padding=11)
        # Line 4, 3x3 maximum convergence layer followed by 1x1 convolution layer
        self.p4_1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv1d(in_channels, c4, kernel_size=1)
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        Inception_out=torch.cat((p1, p2, p3, p4), dim=1)

        return Inception_out
class XrdCNN(nn.Module):
    def __init__(self, tgt=[0]):
        super(XrdCNN, self).__init__()
        # Convolution layers
        self.conv1 = Inception(1, 4, (4, 16), (4, 8), 8)
        self.bn1 = nn.BatchNorm1d(num_features=36)
        self.conv2 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=72)
        self.conv3 = nn.Conv1d(in_channels=72, out_channels=128, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        # Average pooling layer
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        # GasModel
        self.Gasfc1 = nn.Linear(8, 32)  
        self.Gasfc2 = nn.Linear(32, 128)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.15)
        # Dense layers
        self.fc1 = nn.Linear(12288+128, 512)  # Adjusted according to the output size from global average pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # Output layers for regression and classification
        self.out_regression = nn.Linear(128, len(tgt))

    def forward(self, x, gas_input):
        x = x.unsqueeze(1)  # Add a dimension to match Conv1d's input requirements
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(start_dim=1)
        Gas_emb=F.gelu(self.Gasfc1(gas_input))
        Gas_emb=self.Gasfc2(Gas_emb)
        x = torch.cat([x,Gas_emb], dim=-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Separate outputs for regression and classification
        regression_output = self.out_regression(x)
        return regression_output

# 自定义数据集类
class GasAdsorptionDataset(Dataset):
    def __init__(self, material_list, pxrd_data, gas_emb):
        self.samples = []
        # 按材料分组数据
        material_dict = defaultdict(list)
        for item in material_list:
            mat, gas, ads = item
            material_dict[mat].append((gas, ads))
        
        # 构建样本
        for mat, gas_data in material_dict.items():
            pxrd = pxrd_data[mat]['intensity']  # 获取PXRD数据
            for gas, ads in gas_data:
                emb = gas_emb[gas]  # 获取气体描述符
                self.samples.append((pxrd, emb, ads))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pxrd, emb, ads = self.samples[idx]
        return torch.FloatTensor(pxrd).cuda(), \
               torch.FloatTensor(emb).cuda(), \
               torch.FloatTensor([ads]).cuda()

# 留一法交叉验证函数
def material_wise_loo(adinfo_list, pxrd_data, gas_emb, model):
    # 获取所有唯一材料
    materials = list(set([item[0] for item in adinfo_list]))
    
    all_preds = []
    all_true = []
    
    # 遍历每个材料作为测试集
    for test_mat in materials:
        mat_preds = []
        mat_label = []
        # 划分训练测试集
        train_data = [item for item in adinfo_list if item[0] != test_mat]
        test_data = [item for item in adinfo_list if item[0] == test_mat]
        
        # 创建数据集
        train_dataset = GasAdsorptionDataset(train_data, pxrd_data, gas_emb)
        test_dataset = GasAdsorptionDataset(test_data, pxrd_data, gas_emb)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        
        # 克隆新模型
        model = XrdCNN().cuda()  # 创建新实例
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.001
        )
        criterion = nn.MSELoss()
        
        # 训练阶段
        model.train()
        for epoch in range(100):
            for pxrd, emb, ads in train_loader:
                optimizer.zero_grad()
                pred = model(pxrd, emb)
                loss = criterion(pred, ads)
                loss.backward()
                optimizer.step()
        
        # 测试阶段
        model.eval()
        with torch.no_grad():
            for pxrd, emb, ads in test_loader:
                pred = model(pxrd, emb).cpu().numpy()
                all_preds.extend(pred.flatten())
                all_true.extend(ads.cpu().numpy().flatten())
                mat_preds.extend(pred.flatten())
                mat_label.extend(ads.cpu().numpy().flatten())
        print(test_data,list(mat_preds),list(mat_label))
    return np.array(all_preds), np.array(all_true)

# 可视化函数
def plot_performance(true, pred):
    plt.figure(figsize=(8,6))
    plt.scatter(true, pred, alpha=0.6, edgecolors='w')
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--')
    plt.xlabel('True Adsorption', fontsize=12)
    plt.ylabel('Predicted Adsorption', fontsize=12)
    plt.title('Material-wise LOO Validation', fontsize=14)
    
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    plt.text(0.05, 0.9, f'R² = {r2:.3f}\nMAE = {mae:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.show()

# 主程序
if __name__ == "__main__":
    model=XrdCNN().cuda()
    [adinfo_list, pxrd_data, ads_gasEMB]=torch.load('data-up/ZIF_PXRD,AD.pt')
    predictions, true_values = material_wise_loo(
        adinfo_list, pxrd_data, ads_gasEMB, model
    )
    plot_performance(true_values, predictions)