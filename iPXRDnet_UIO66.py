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
    def __init__(self, tgt):
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
        # Dropout layer
        self.dropout = nn.Dropout(p=0.15)
        # Dense layers
        self.fc1 = nn.Linear(12288, 512)  # Adjusted according to the output size from global average pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # Output layers for regression and classification
        self.out_regression = nn.Linear(128, len(tgt))

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a dimension to match Conv1d's input requirements
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Separate outputs for regression and classification
        regression_output = self.out_regression(x)
        return regression_output

# 自定义数据集类
class PXRDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features).cuda()
        self.labels = torch.FloatTensor(labels).cuda()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 留一法交叉验证函数
def leave_one_out_validation(PXRD_AD, tgt):
    all_preds = []
    all_labels = []
    material_names = list(PXRD_AD.keys())
    
    # 数据预处理
    features = np.array([v[:801] for v in PXRD_AD.values()])
    labels = np.array([v[-1] for v in PXRD_AD.values()])
    
    # 遍历所有样本
    for i in range(len(material_names)):
        # 划分训练集和测试集
        train_idx = [j for j in range(len(material_names)) if j != i]
        test_idx = [i]
        
        # 初始化新模型
        model = XrdCNN(tgt).cuda()  # 假设tgt参数已定义
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=5e-5,
            weight_decay=0.001  # 确保已定义该参数
        )
        
        # 创建数据加载器
        train_dataset = PXRDataset(features[train_idx], labels[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
        
        # 训练模型
        model.train()
        for epoch in range(100):  # 训练epoch数可根据需要调整
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), targets)
                loss.backward()
                optimizer.step()
        
        # 验证模型
        model.eval()
        with torch.no_grad():
            test_input = torch.FloatTensor(features[test_idx]).cuda()
            pred = model(test_input).cpu().numpy()[0]
            
        all_preds.append(pred)
        all_labels.append(labels[test_idx][0])
        print(material_names[test_idx[0]],pred[0],labels[test_idx][0])
    
    return np.array(all_preds), np.array(all_labels)

# 结果可视化函数
def plot_results(true, pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, alpha=0.6)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('LOO Validation Performance')
    
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}', 
             transform=plt.gca().transAxes, va='top')
    
    plt.show()

# 主程序
if __name__ == "__main__":
    tgt=[0]
    PXRD_AD=torch.load('data-up/Uio-66_PXRD_AD.pt')
    predictions, true_values = leave_one_out_validation(PXRD_AD, tgt)
    plot_results(true_values, predictions)