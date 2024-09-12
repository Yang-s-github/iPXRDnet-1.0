#def
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
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
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from time import time, strftime, localtime
from torch.nn.functional import   relu
import csv,json,copy,torch,re
from sklearn.model_selection import KFold

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
        self.fc1 = nn.Linear(26624, 512)  
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
#data 
data = torch.load('data/MOD_XRD_dict.pt')
data_y_all=torch.load('data/MOD_data.pt')
INFOdata={}
for item in data_y_all:
    try:
        INFOdata[item[0]]=list(map(float, item[1:]))
    except:
        print(item)
adinfo_list_new=[]
for i in data.keys():
    if i in INFOdata.keys() and i.startswith( 'mof_' ):
        xrd_ad_str_info=torch.tensor(INFOdata[i]+data[i])
        adinfo_list_new.append([i,xrd_ad_str_info])
print(len(adinfo_list_new))
del data,data_y_all
tgt = [1,3,4,6,14, 15]
target_dict = ['Di','Df','Dif','rho','VSA','GSA','VPOV','GPOV','POAV_vol_frac','PONAV_vol_frac','GPOAV','GPONAV','POAV','PONAV','KVRH','GVRH']
target_dict_selected = []
for i in range(len(tgt)):
    target_dict_selected.append(target_dict[tgt[i]])
print(target_dict_selected)

train_ratio=0.8
seednum=9764
a=np.arange(len(adinfo_list_new))

np.random.seed(seednum)
torch.manual_seed(seednum)
torch.cuda.manual_seed_all(seednum)
random.seed(seednum)
np.random.shuffle(a)
train_idx=a[:int(len(a)*train_ratio)]
test_idx=a[int(len(a)*train_ratio):]
print(a[1:10],len(adinfo_list_new))
set_train = [adinfo_list_new[i] for i in train_idx]
set_test = [adinfo_list_new[i] for i in test_idx]
data_y_train=[]
for InputInfo in set_train:
    key=InputInfo[0]
    data_y_train.append(InputInfo[1])
tensor_input_train=torch.stack(data_y_train)
data_y_train=tensor_input_train[:,1701:]
data_y_train=data_y_train[:, tgt]
scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(data_y_train.shape[-1])]



torch.cuda.set_device(0)
device = torch.device("cuda")
epochs = 120
print("epoch\t loss\t")
bestmse=10000
t1 = time()
batch_size_train=16
loss_func = nn.MSELoss()
multiple_train_steps = 0.1
multiple_weight_decay = 0.001


my_re_manual_seed=42
kf = KFold(n_splits=5, shuffle=True, random_state=my_re_manual_seed)
all_test_r2,all_test_mae=[],[]

dataset=set_train+set_test
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Fold :{fold + 1}")
    
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size_train, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=400, sampler=test_subsampler)
    epochs = 120
    train_size = len(train_idx)
    print("epoch\t loss\t")
    bestmse=10000
    t1 = time()
    batch_size_train=16
    loss_func = nn.MSELoss()
    multiple_train_steps = 0.1
    multiple_weight_decay = 0.001

    loss_func = nn.MSELoss()
    model=XrdCNN(tgt).cuda()
    num_train_optimization_steps = int(train_size/batch_size_train * epochs)
    optim = torch.optim.AdamW(model.parameters(), lr=5.0e-5, weight_decay=multiple_weight_decay)
    lr_scheduler =  transformers.get_linear_schedule_with_warmup(optim,int(num_train_optimization_steps *multiple_train_steps),num_train_optimization_steps)
    for epoch in range(epochs):
        loss = 0.0
        model.train()
        train_label=[]
        train_pred=[]
        for item in train_loader:
            data_xrd=item[1][:,:1701]
            data_xrd = data_xrd.to(device)
            unit_y=item[1][:,1701:]
            unit_y=unit_y[:, tgt]
            for i in range(unit_y.shape[-1]):
                unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
            labels=unit_y.to(device)
            y_hat = model(data_xrd)
            train_label.append(labels)
            train_pred.append(y_hat)
            loss_batch = loss_func(y_hat,labels)
            loss = loss+loss_batch.item()
            optim.zero_grad()
            loss_batch.backward()
            optim.step()
            lr_scheduler.step()
        if((epoch+1)%1000 == 0):
            train_label = torch.cat(train_label).cuda()
            train_pred = torch.cat(train_pred).cuda()
            print(epoch,optim.param_groups[0]['lr'] * 1e5)
            for i in range(0,len(tgt),1):
                train_pred[:, i] = train_pred[:, i] * scales[i][1] + scales[i][0]
                train_label[:, i] = train_label[:, i] * scales[i][1] + scales[i][0]
                r2 = round(r2_score(train_label[:, i].detach().cpu().numpy(), train_pred[:, i].detach().cpu().numpy()), 3)
                print(f'train {target_dict[tgt[i]]} R2: {r2}')
            print("time: {:.5f}".format(time() - t1))
            t1 = time()
        mse=0
    model.eval()
    test_label=[]
    test_pred=[]
    mse=0
    with torch.no_grad():
        for item in test_loader: 
            data_xrd=item[1][:,:1701]
            data_xrd = data_xrd.to(device)
            unit_y=item[1][:,1701:]
            unit_y=unit_y[:, tgt]
            for i in range(unit_y.shape[-1]):
                unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
            labels=unit_y.to(device)
            y_hat = model(data_xrd)
            mse += loss_func(y_hat, labels).item()
            test_label.append(labels)
            test_pred.append(y_hat)
    fold_test_r2 = []
    fold_test_mae = []
    test_label = torch.cat(test_label).cuda()
    test_pred = torch.cat(test_pred).cuda()
    for i in range(len(tgt)):
        test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
        test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
        r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
        mae = mean_absolute_error(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy())
        print(f'dev {target_dict[tgt[i]]} R2: {r2}')
        fold_test_r2.append(r2)
        fold_test_mae.append(mae)
    print("{}\t dev mse:{:.3f} r2 {:.5f}".format(i+1,mse,r2.item()))
        
    all_test_r2.append(fold_test_r2)
    all_test_mae.append(fold_test_mae)

mean_r2 = np.mean(all_test_r2, axis=0)
mean_mae = np.mean(all_test_mae, axis=0)
for i, tgt_name in enumerate(tgt):
    print(f'Average {target_dict[tgt_name]} R2: {mean_r2[i]:.3f}, MAE: {mean_mae[i]:.3f}')
torch.save(model.state_dict(),'MOD-8:2.pt')
