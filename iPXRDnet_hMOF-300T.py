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

adinfo_list_new = torch.load("data/hmof-xrd+str+ad.pt")
tgt = [0,3,6,13,16,18,19,20,21,22,23]
target_dict = ['CO2_uptake_P0.15bar_T298K [mmol/g]','heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]','excess_CO2_uptake_P0.15bar_T298K [mmol/g]','CO2_uptake_P0.10bar_T363K [mmol/g]','heat_adsorption_CO2_P0.10bar_T363K [kcal/mol]','excess_CO2_uptake_P0.10bar_T363K [mmol/g]','CO2_uptake_P0.70bar_T413K [mmol/g]','heat_adsorption_CO2_P0.70bar_T413K [kcal/mol]','excess_CO2_uptake_P0.70bar_T413K [mmol/g]','working_capacity_vacuum_swing [mmol/g]','working_capacity_temperature_swing [mmol/g]','CO2_binary_uptake_P0.15bar_T298K [mmol/g]','heat_adsorption_CO2_binary_P0.15bar_T298K [kcal/mol]','excess_CO2_binary_uptake_P0.15bar_T298K [mmol/g]','N2_binary_uptake_P0.85bar_T298K [mmol/g]','heat_adsorption_N2_binary_P0.85bar_T298K [kcal/mol]','excess_N2_binary_uptake_P0.85bar_T298K [mmol/g]','CO2/N2_selectivity','surface_area [m^2/g]','void_fraction','void_volume [cm^3/g]','largest_free_sphere_diameter [A]','largest_included_sphere_along_free_sphere_path_diameter [A]','largest_included_sphere_diameter [A]']
target_dict_selected=[]
for i in range(len(tgt)):
    target_dict_selected.append(target_dict[tgt[i]])
print(target_dict_selected)

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15
seedNUM=9764
np.random.seed(seedNUM)
torch.manual_seed(seedNUM)
torch.cuda.manual_seed_all(seedNUM)
random.seed(seedNUM)
a=np.arange(len(adinfo_list_new))
np.random.shuffle(a)
train_idx=a[:int(len(a)*train_ratio)]
val_idx=a[int(len(a)*train_ratio):-int(len(a)*test_ratio)]
test_idx=a[-int(len(a)*test_ratio):]
set_train = [adinfo_list_new[i] for i in train_idx]
set_val = [adinfo_list_new[i] for i in val_idx]
set_test = [adinfo_list_new[i] for i in test_idx]

data_y_train=[]
for InputInfo in set_train:
    key=InputInfo[0]
    data_y_train.append(InputInfo[1])
tensor_input_train=torch.stack(data_y_train)
data_y_train=tensor_input_train[:,1701:]
data_y_train=data_y_train[:, tgt]
scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(data_y_train.shape[-1])]
print(scales)

torch.cuda.set_device(0)
device = torch.device("cuda")
epochs = 120
print("epoch\t loss\t")
bestmse=10000
t1 = time()
batch_size_train=256
loss_func = nn.MSELoss()
multiple_train_steps = 0.1
multiple_weight_decay = 0.001

train_loader = DataLoader(set_train, batch_size=batch_size_train, shuffle=True)
dev_loader = DataLoader(set_val, batch_size=256)
test_loader = DataLoader(set_test, batch_size=256)

train_size=int(len(set_train)/batch_size_train)
total_steps=int(train_size * epochs)
print(total_steps,train_size,len(set_train))

#run
model=XrdCNN(tgt).cuda()
num_train_optimization_steps = int(train_size * epochs)
optim = torch.optim.AdamW(model.parameters(), lr=5.0e-5, weight_decay=multiple_weight_decay)
lr_scheduler =  transformers.get_linear_schedule_with_warmup(optim,int(num_train_optimization_steps *multiple_train_steps),num_train_optimization_steps)
bestmse=100000
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
    if((epoch+1)%40 == 0):
        train_label = torch.cat(train_label).cuda()
        train_pred = torch.cat(train_pred).cuda()
        print(epoch,optim.param_groups[0]['lr'] * 1e5)
        for i in range(0,len(tgt),1):
            train_pred[:, i] = train_pred[:, i] * scales[i][1] + scales[i][0]
            train_label[:, i] = train_label[:, i] * scales[i][1] + scales[i][0]
            r2 = round(r2_score(train_label[:, i].detach().cpu().numpy(), train_pred[:, i].detach().cpu().numpy()), 3)
            print(f'Train {target_dict[tgt[i]]} R2: {r2}')
        print("time: {:.5f}".format(time() - t1))
        t1 = time()
        
        test_label = []
        test_pred = []
        mse=0
        best_model.eval()
        with torch.no_grad():
            for item in dev_loader:
                data_xrd=item[1][:,:1701]
                data_xrd = data_xrd.to(device)
                unit_y=item[1][:,1701:]
                unit_y=unit_y[:, tgt]
                for i in range(unit_y.shape[-1]):
                    unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
                labels=unit_y.to(device)
                y_hat = best_model(data_xrd)
                mse += loss_func(y_hat, labels).item()
                test_label.append(labels)
                test_pred.append(y_hat)
        
        test_label = torch.cat(test_label).cuda()
        test_pred = torch.cat(test_pred).cuda()
        for i in range(0,len(tgt),1):
            test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
            test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
            r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
            mae = mean_absolute_error(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy())
            print(f'Dev {target_dict[tgt[i]]} R2: {r2:.3f}, MAE: {mae:.3f}')
        print("{}\t Dev mse:{:.3f} r2 {:.5f}".format(i+1,mse,r2.item()))
        
        
    mse=0
    test_label = []
    test_pred = []
    model.eval()
    with torch.no_grad():
        for item in dev_loader:
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
            
    if mse < bestmse:
        best_model=copy.deepcopy(model)
        bestmse=mse

        
best_model.eval()
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
        y_hat = best_model(data_xrd)
        mse += loss_func(y_hat, labels).item()
        test_label.append(labels)
        test_pred.append(y_hat)

test_label = torch.cat(test_label).cuda()
test_pred = torch.cat(test_pred).cuda()
for i in range(0,len(tgt),1):
    test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
    test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
    r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
    mae = mean_absolute_error(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy())
    print(f'Test {target_dict[tgt[i]]} R2: {r2:.3f}, MAE: {mae:.3f}')
print("{}\t Test mse:{:.3f} r2 {:.5f}".format(i+1,mse,r2.item()))
torch.save(best_model.state_dict(),'hmof-300T.pt')

