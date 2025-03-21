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
from scipy.interpolate import interp1d


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
    def __init__(self, tgt,dim_OF):
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
        self.fc1 = nn.Linear(dim_OF, 2048)  # Adjusted according to the output size from global average pooling
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        # Output layers for regression and classification
        self.out_regression = nn.Linear(256, len(tgt))

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
    
def dim_of_flatten(data_xrd):
    # 卷积层
    Test_Shape_conv1 = Inception(1, 4, (4, 16), (4, 8), 8)
    Test_Shape_bn1 = nn.BatchNorm1d(num_features=36)
    Test_Shape_conv2 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=5, stride=1)
    Test_Shape_bn2 = nn.BatchNorm1d(num_features=72)
    Test_Shape_conv3 = nn.Conv1d(in_channels=72, out_channels=128, kernel_size=5, stride=1)
    Test_Shape_bn3 = nn.BatchNorm1d(num_features=128)
    Test_Shape_conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
    Test_Shape_bn4 = nn.BatchNorm1d(num_features=256)
    Test_Shape_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    x = data_xrd.float().cpu().unsqueeze(1)  
    x = Test_Shape_pool(F.relu(Test_Shape_bn1(Test_Shape_conv1(x))))
    x = Test_Shape_pool(F.relu(Test_Shape_bn2(Test_Shape_conv2(x))))
    x = Test_Shape_pool(F.relu(Test_Shape_bn3(Test_Shape_conv3(x))))
    x = Test_Shape_pool(F.relu(Test_Shape_bn4(Test_Shape_conv4(x))))
    x = x.flatten(start_dim=1)
    dim_out=x.shape[1]
    return dim_out

def xrd_to_new_step(data_xrd,step_size):
    original_angles = np.linspace(5, 45, 801)
    original_intensities = data_xrd[:801].numpy() 
    interpolator = interp1d(original_angles, original_intensities)
    new_angles = np.arange(5, 45, step_size)
    new_intensities = interpolator(new_angles)
    new_intensities = torch.tensor(new_intensities)
    return new_intensities


all_adinfo_list_new=torch.load('data/all_adinfo_list_robustness.pt')
tgt = [1,5,7,8]
target_dict = ['Di','Df','Dif','cm3_g','ASA_m^2/cm^3','ASA_m2_g','AV_VF','AV_cm3_g','Gas uptake']
target_dict_selected = []
for i in range(len(tgt)):
    target_dict_selected.append(target_dict[tgt[i]])
print(target_dict_selected)
STR_DICT=torch.load('data/hMOF-130T_STR_DICT.pt')
GAS_DICT=torch.load('data/hMOF-130T_GAS_DICT.pt')
ORXRD_DICT=torch.load('data/hmof-130T_50nm.pt')

# Step size 
step_size_list=[0.05,0.075,0.10,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5]
for step_size in step_size_list:
    tgt = [1,5,7,8,9]
    target_dict = ['Di','Df','Dif','cm3_g','ASA_m^2/cm^3','ASA_m2_g','AV_VF','AV_cm3_g','CO2 uptake 50kpa','CH4 uptake 50kpa']
    target_dict_selected = []
    for i in range(len(tgt)):
        target_dict_selected.append(target_dict[tgt[i]])
    print(target_dict_selected,'RUNNING step size',step_size)
    adinfo_list_new=[]
    for item in all_adinfo_list_new:
        new_intensities=torch.tensor(xrd_to_new_step(ORXRD_DICT[item[0]],step_size)).float()
        newxrd_ad=torch.cat([new_intensities,item[1]],dim=0)
        adinfo_list_new.append([item[0],newxrd_ad])
    new_intensities=xrd_to_new_step(ORXRD_DICT[item[0]],step_size).float()
    stack_new_intensities=torch.stack([new_intensities,new_intensities],dim=0)
    dim_OF=dim_of_flatten(stack_new_intensities)
    XRD_lenght=len(new_intensities)

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
    data_y_train=tensor_input_train[:,XRD_lenght:]
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
    model=XrdCNN(tgt,dim_OF).cuda()
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
            data_xrd=item[1][:,:XRD_lenght]
            data_xrd = data_xrd.to(device)
            unit_y=item[1][:,XRD_lenght:]
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
                    data_xrd=item[1][:,:XRD_lenght]
                    data_xrd = data_xrd.to(device)
                    unit_y=item[1][:,XRD_lenght:]
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
                data_xrd=item[1][:,:XRD_lenght]
                data_xrd = data_xrd.to(device)
                unit_y=item[1][:,XRD_lenght:]
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

        mse=0

    best_model.eval()
    test_label=[]
    test_pred=[]
    mse=0
    with torch.no_grad():
        for item in test_loader: 
            data_xrd=item[1][:,:XRD_lenght]
            data_xrd = data_xrd.to(device)
            unit_y=item[1][:,XRD_lenght:]
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
    savename='hmof-130T-0.15dp-120ep_'+'Step_Size'+str(step_size)+'.pt'
    torch.save(best_model.state_dict(),savename)
    del best_model,model,bestmse

# Redefine the XrdCNN model

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
        self.fc1 = nn.Linear(12288, 2048)  # Adjusted according to the output size from global average pooling
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        # Output layers for regression and classification
        self.out_regression = nn.Linear(256, len(tgt))

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

# XRD size
XRD_size_list=['hmof-130T_1nm','hmof-130T_5nm','hmof-130T_10nm','hmof-130T_15nm','hmof-130T_25nm','hmof-130T_50nm','hmof-130T_75nm','hmof-130T_100nm']
for  XRD_size in XRD_size_list:
    XRD_DICT=torch.load('data/'+XRD_size+'.pt')
    print(XRD_size)
    

    tgt = [1,5,7,8,9]
    target_dict = ['Di','Df','Dif','cm3_g','ASA_m^2/cm^3','ASA_m2_g','AV_VF','AV_cm3_g','CO2 uptake 50kpa','CH4 uptake 50kpa']
    target_dict_selected = []
    for i in range(len(tgt)):
        target_dict_selected.append(target_dict[tgt[i]])
    adinfo_list_new=[]
    for item in all_adinfo_list_new:
        new_intensities=torch.tensor(XRD_DICT[item[0]]).float()
        newxrd_ad=torch.cat([new_intensities,item[1]],dim=0)
        adinfo_list_new.append([item[0],newxrd_ad])
        
    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15
    seedNUM=1234
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
    data_y_train=tensor_input_train[:,801:]
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
            data_xrd=item[1][:,:801]
            data_xrd = data_xrd.to(device)
            unit_y=item[1][:,801:]
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
                    data_xrd=item[1][:,:801]
                    data_xrd = data_xrd.to(device)
                    unit_y=item[1][:,801:]
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
                data_xrd=item[1][:,:801]
                data_xrd = data_xrd.to(device)
                unit_y=item[1][:,801:]
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

        mse=0

    best_model.eval()
    test_label=[]
    test_pred=[]
    mse=0
    with torch.no_grad():
        for item in test_loader: 
            data_xrd=item[1][:,:801]
            data_xrd = data_xrd.to(device)
            unit_y=item[1][:,801:]
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
    savename='hmof-130T-120E-size_'+str(XRD_size)+'.pt'
    torch.save(best_model.state_dict(),savename)
    del best_model,model,bestmse

#Noise

    
def add_noise_to_tensor(tensor,maxnoise,Lnoise):
    noise = torch.tensor(np.random.uniform(0, maxnoise, size=tensor.shape))
    low_values = np.where(tensor < Lnoise)
    tensor[low_values] += noise[low_values]
    return tensor


maxnoise_list=[1,2,3,4]
Lnoise_list=[3,4,5]
for maxnoise in maxnoise_list:
    for Lnoise in Lnoise_list:
        
        tgt = [1,5,7,8,9]
        target_dict = ['Di','Df','Dif','cm3_g','ASA_m^2/cm^3','ASA_m2_g','AV_VF','AV_cm3_g','CO2 uptake 50kpa','CH4 uptake 50kpa']
        target_dict_selected = []
        for i in range(len(tgt)):
            target_dict_selected.append(target_dict[tgt[i]])
        adinfo_list_new=[]
        for item in all_adinfo_list_new:
            new_intensities=torch.tensor(add_noise_to_tensor(ORXRD_DICT[item[0]],maxnoise,Lnoise)).float()
            newxrd_ad=torch.cat([new_intensities,item[1]],dim=0)
            adinfo_list_new.append([item[0],newxrd_ad])

        for i in range(len(tgt)):
            target_dict_selected.append(target_dict[tgt[i]])
        train_ratio=0.7
        val_ratio=0.15
        test_ratio=0.15
        seedNUM=1234
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
        data_y_train=tensor_input_train[:,801:]
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
                data_xrd=item[1][:,:801]
                data_xrd = data_xrd.to(device)
                unit_y=item[1][:,801:]
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
                        data_xrd=item[1][:,:801]
                        data_xrd = data_xrd.to(device)
                        unit_y=item[1][:,801:]
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
                    data_xrd=item[1][:,:801]
                    data_xrd = data_xrd.to(device)
                    unit_y=item[1][:,801:]
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

            mse=0

        best_model.eval()
        test_label=[]
        test_pred=[]
        mse=0
        with torch.no_grad():
            for item in test_loader: 
                data_xrd=item[1][:,:801]
                data_xrd = data_xrd.to(device)
                unit_y=item[1][:,801:]
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
        savename='hmof-130T-120E_'+'MaxNoise_'+str(maxnoise)+'_LNoise_'+str(Lnoise)+'.pt'
        torch.save(best_model.state_dict(),savename)
        del best_model,model,bestmse
