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
import csv,json,copy,torch,re,math

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
    def __init__(self, tgt,min_max_pressure):
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
        self.Gasfc1 = nn.Linear(6, 32)  
        self.Gasfc2 = nn.Linear(32, 128)
        # PressureModel
        self.Pfc1 = nn.Linear(1, 32)  
        self.Pfc2 = nn.Linear(32, 128)
        self.pressure_embed = nn.Embedding(32, 128)
        self.min_max_P=min_max_pressure
        # Dropout layer
        self.dropout = nn.Dropout(p=0.15)
        # Dense layers
        self.fc1 = nn.Linear(26624+128*3, 2048)  # Adjusted according to the output size from global average pooling
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        # Output layers for regression and classification
        self.out_regression = nn.Linear(256, len(tgt))

    def forward(self, x, gas_input, pressure):
        x = x.unsqueeze(1)  # Add a dimension to match Conv1d's input requirements
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(start_dim=1)
        
        Gas_emb=F.gelu(self.Gasfc1(gas_input))
        Gas_emb=self.Gasfc2(Gas_emb)
        pressure = torch.clamp(pressure, self.min_max_P[0], self.min_max_P[1])
        pressure = (pressure - self.min_max_P[0]) / (self.min_max_P[1] - self.min_max_P[0])
        
        Pressure_emb=F.gelu(self.Pfc1(pressure.unsqueeze(1)))
        Pressure_emb=self.Pfc2(Pressure_emb)
        
        pressure_bin = torch.floor(pressure * 32).to(torch.long)
        Pressure_emb2=self.pressure_embed(pressure_bin)
        
        x = torch.cat([x,Pressure_emb], dim=-1)
        x = torch.cat([x,Pressure_emb2], dim=-1)
        x = torch.cat([x,Gas_emb], dim=-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        # Separate outputs for regression and classification
        regression_output = self.out_regression(x)
        return regression_output

tgt = [1,5,7,8]
target_dict = ['Di','Df','Dif','cm3_g','ASA_m^2/cm^3','ASA_m2_g','AV_VF','AV_cm3_g','Gas uptake']
target_dict_selected = []
for i in range(len(tgt)):
    target_dict_selected.append(target_dict[tgt[i]])
print(target_dict_selected)
XRD_DICT=torch.load('data/hMOF-130T_XRD_DICT.pt')
STR_DICT=torch.load('data/hMOF-130T_STR_DICT.pt')
GAS_DICT=torch.load('data/hMOF-130T_GAS_DICT.pt')

GAS_DICT['H2']=GAS_DICT['Hydrogen']
GAS_DICT['N2']=GAS_DICT['Nitrogen']
GAS_DICT['C1']=GAS_DICT['Methane']
GAS_DICT['CO2']=GAS_DICT['CarbonDioxide']

for TG_list in  [['H2', '100000'],['N2', '100000'],['CO2', '100000'],['C1', '100000'],['H2', '1000'],['CO2', '1000'],['N2', '1000'],['C1', '1000']]:
    print(TG_list)
    [set_train,set_val,set_test]=torch.load('data-up/4gas_'+TG_list[0]+'_'+TG_list[1]+'.pt')

    list_pressure=[]
    list_gas=[]
    for InputInfo in set_train:
        pressure=float(InputInfo[2])
        list_gas.append(InputInfo[1])
        list_pressure.append(math.log10(pressure))
    set_P=set(list_pressure)
    set_gas=set(list_gas)
    minimum_list_P = min(set_P)-1
    maximum_list_P = max(set_P)+1
    min_max_pressure=[minimum_list_P,maximum_list_P]
    data_y_train=[]
    for InputInfo in set_train:
        key=InputInfo[0]
        ad_data=float(InputInfo[3])
        ad_str=torch.cat([STR_DICT[key],torch.tensor([ad_data])],dim=0)
        data_y_train.append(ad_str)
    tensor_input_train=torch.stack(data_y_train)
    data_y_train=tensor_input_train[:, tgt]
    scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(data_y_train.shape[-1])]
    print(scales)

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    epochs = 60
    bestmse=1e30
    t1 = time()
    t2 = time()
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

    model=XrdCNN(tgt,min_max_pressure).cuda()
    num_train_optimization_steps = int(train_size * epochs)
    optim = torch.optim.AdamW(model.parameters(), lr=10.0e-5, weight_decay=multiple_weight_decay)
    lr_scheduler =  transformers.get_linear_schedule_with_warmup(optim,int(num_train_optimization_steps *multiple_train_steps),num_train_optimization_steps)
    #run
    for epoch in range(epochs):
        loss = 0.0
        model.train()
        train_label=[]
        train_pred=[]
        for item in train_loader:
            Xrd_list,ADSTR_list,Gas_list,Pressure_list=[],[],[],[]
            for indexnum in range(len(item[0])):
                XRD_INFO=XRD_DICT[item[0][indexnum]]
                Xrd_list.append(XRD_INFO)

                ad_data=float(item[-1][indexnum])
                ADSTR_INFO=torch.cat([STR_DICT[item[0][indexnum]],torch.tensor([ad_data])],dim=0)
                ADSTR_list.append(ADSTR_INFO)
                GAS_INFO=GAS_DICT[item[1][indexnum]]
                Gas_list.append(GAS_INFO)
                PRESSURE_INFO=math.log10(float(item[2][indexnum]))
                Pressure_list.append(PRESSURE_INFO)
            data_XRD=torch.stack(Xrd_list).to(device)
            ADSTR_list=torch.stack(ADSTR_list)
            labels=ADSTR_list[:, tgt].to(device)
            for i in range(labels.shape[-1]):
                labels[:, i] = (labels[:, i] - scales[i][0]) / scales[i][1]
            Pressure_input=torch.tensor(Pressure_list).to(device)
            Gas_input=torch.tensor(Gas_list).to(device)

            y_hat = model(data_XRD,Gas_input,Pressure_input)
            train_label.append(labels)
            train_pred.append(y_hat)
            loss_batch = loss_func(y_hat,labels)
            loss = loss+loss_batch.item()
            optim.zero_grad()
            loss_batch.backward()
            optim.step()
            lr_scheduler.step()


        mse=0
        test_label = []
        test_pred = []
        model.eval()
        with torch.no_grad():
            for item in dev_loader:
                Xrd_list,ADSTR_list,Gas_list,Pressure_list=[],[],[],[]
                for indexnum in range(len(item[0])):
                    XRD_INFO=XRD_DICT[item[0][indexnum]]
                    Xrd_list.append(XRD_INFO)

                    ad_data=float(item[-1][indexnum])
                    ADSTR_INFO=torch.cat([STR_DICT[item[0][indexnum]],torch.tensor([ad_data])],dim=0)
                    ADSTR_list.append(ADSTR_INFO)
                    GAS_INFO=GAS_DICT[item[1][indexnum]]
                    Gas_list.append(GAS_INFO)
                    PRESSURE_INFO=math.log10(float(item[2][indexnum]))
                    Pressure_list.append(PRESSURE_INFO)
                data_XRD=torch.stack(Xrd_list).to(device)
                ADSTR_list=torch.stack(ADSTR_list)
                labels=ADSTR_list[:, tgt].to(device)
                for i in range(labels.shape[-1]):
                    labels[:, i] = (labels[:, i] - scales[i][0]) / scales[i][1]
                Pressure_input=torch.tensor(Pressure_list).to(device)
                Gas_input=torch.tensor(Gas_list).to(device)

                y_hat = model(data_XRD,Gas_input,Pressure_input)
                mse += loss_func(y_hat[:, -1], labels[:, -1]).item()
                test_label.append(labels)
                test_pred.append(y_hat)

        if mse < bestmse:
            best_model=copy.deepcopy(model)
            bestmse=mse

        mse=0

    best_model.eval()
    test_label=[]
    test_pred=[]

    predictions = {}

    with torch.no_grad():
        for item in test_loader: 
            Xrd_list,ADSTR_list,Gas_list,Pressure_list=[],[],[],[]
            for indexnum in range(len(item[0])):
                XRD_INFO=XRD_DICT[item[0][indexnum]]
                Xrd_list.append(XRD_INFO)

                ad_data=float(item[-1][indexnum])
                ADSTR_INFO=torch.cat([STR_DICT[item[0][indexnum]],torch.tensor([ad_data])],dim=0)
                ADSTR_list.append(ADSTR_INFO)

                GAS_INFO=GAS_DICT[item[1][indexnum]]
                Gas_list.append(GAS_INFO)
                PRESSURE_INFO=math.log10(float(item[2][indexnum]))
                Pressure_list.append(PRESSURE_INFO)
            data_XRD=torch.stack(Xrd_list).to(device)
            ADSTR_list=torch.stack(ADSTR_list)
            labels=ADSTR_list[:, tgt].to(device)
            for i in range(labels.shape[-1]):
                labels[:, i] = (labels[:, i] - scales[i][0]) / scales[i][1]
            Pressure_input=torch.tensor(Pressure_list).to(device)
            Gas_input=torch.tensor(Gas_list).to(device)

            y_hat = best_model(data_XRD,Gas_input,Pressure_input)
            mse += loss_func(y_hat[:,-1], labels[:,-1]).item()
            test_label.append(labels)
            test_pred.append(y_hat)

    test_label = torch.cat(test_label).to(device)
    test_pred = torch.cat(test_pred).to(device)
    for i in range(0,len(tgt),1):
        test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
        test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
        r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
        mae = mean_absolute_error(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy())
        print(f'Test {target_dict[tgt[i]]} R2: {r2:.3f}, MAE: {mae:.3f}')

