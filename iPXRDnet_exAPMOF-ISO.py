#def
from sklearn.metrics import  r2_score
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from torch import nn
from time import time
import json,copy,torch

from script_for_models.parsing import parse_train_args, modify_train_args
from script_for_models.model import  get_cmpn_encoder,encoder_add_functional_prompt 

def process_CIFNAME(CIFNAME):
    spname=CIFNAME.split('-')
    if CIFNAME != 'SIFSIX-pyz-SH-Ni':
        A=spname[0]
        L=spname[1]
        M=spname[2]
    else:
        A='SIFSIX'
        L='pyz-SH'
        M='Ni'
    if  'FSIX' not in  A:
        if 'BSF' in A:
            if L in L_Reflection['BSF'].keys():
                L=L_Reflection['BSF'][L]
        elif 'ALFFIVE' in A:
            if L in L_Reflection['ALFFIVE'].keys():
                L=L_Reflection['ALFFIVE'][L]
        elif 'DICRO' in A:
            if L in L_Reflection['DICRO'].keys():
                L=L_Reflection['DICRO'][L]
        elif 'OFFIVE' in A:
            if L in L_Reflection['OFFIVE'].keys():
                L=L_Reflection['OFFIVE'][L]
        elif 'OFOUR' in A:
            if L in L_Reflection['OFOUR'].keys():
                L=L_Reflection['OFOUR'][L]
    return A,L,M

def process_InputInfo(InputInfo):
    gas_item=gas_dict[InputInfo[1]]
    A,L,M=process_CIFNAME(InputInfo[0])
    A_item=A_dict[A]
    L_item=L_dict[L]
    M_item=M_dict[M]
    data_y=[float(InputInfo[3])]
    data_P=math.log10(float(InputInfo[2]))
    data_xrd=DICT_XRD[InputInfo[0]]
    return data_xrd,data_y,A_item,L_item,M_item,gas_item,data_P

def process_dataY(data_y_list):
    data_y_array = np.array(data_y_list)
    newy=[]
    for i1 in data_y_array:

        a_float_m = map(float, i1)
        a_float_m = list(a_float_m)
        newy.append(a_float_m)
    data_y_tensor=torch.tensor(newy)
    return data_y_tensor
def tran_list(item):
    transformed_list = []
    for i in range(len(item[0])):
        new_sublist = []
        for new_index in range(len(item)):
            new_sublist.append(item[new_index][i])
        transformed_list.append(new_sublist)
    return transformed_list

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

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a dimension to match Conv1d's input requirements
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(start_dim=1)
        return x

        
class RegressionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        inner_dim,
        tgt,
        last_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)  
        self.dropout = nn.Dropout(p=last_dropout)  
        self.fc1 = nn.Linear(input_dim, inner_dim)  
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out_regression = nn.Linear(256, tgt)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        regression_output = self.out_regression(x)

        return regression_output
    
class MetalModel(nn.Module):
    def __init__(self, input_dim=15, gas_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)  
        self.fc2 = nn.Linear(32, 128)

    def forward(self, input_x): 
        input_x=F.gelu(self.fc1(input_x))
        output_x=self.fc2(input_x)
        return output_x
        
class PressureModel(nn.Module):
    def __init__(self, min_max_pressure):
        super().__init__()
        self.Pfc1 = nn.Linear(1, 32)  
        self.Pfc2 = nn.Linear(32, 128)
        self.pressure_embed = nn.Embedding(128, 512)
        self.min_max_P=min_max_pressure

    def forward(self, pressure): 
        pressure = torch.clamp(pressure, self.min_max_P[0], self.min_max_P[1])
        pressure = (pressure - self.min_max_P[0]) / (self.min_max_P[1] - self.min_max_P[0])
        
        Pressure_emb=F.gelu(self.Pfc1(pressure.unsqueeze(1)))
        Pressure_emb=self.Pfc2(Pressure_emb)
        
        pressure_bin = torch.floor(pressure * 128).to(torch.long)
        Pressure_emb2=self.pressure_embed(pressure_bin)
        
        output_x = torch.cat([Pressure_emb,Pressure_emb2], dim=-1)
        return output_x
        
def build_model(tgt,min_max_pressure,embed_dim=1024, dropout=0.05):
    model = Encoder3D(XrdCNN(tgt=tgt), RegressionHead(embed_dim+300*3+128*6,2048, len(tgt), last_dropout=dropout),min_max_pressure=min_max_pressure)
    return model
    
def collate_fn(data):
    unit_xrd=[]
    unit_y=[]
    unit_name=[]
    anion_list,ligand_list,M_list,gas_Smile_list,P_list=[],[],[],[],[]
    for unit in data:
        unit_xrd.append(unit[0])
        unit_y.append(unit[1])
        unit_name.append(unit[2])
        anion_list.append(unit[3])
        ligand_list.append(unit[4])
        M_list.append(unit[5])
        gas_Smile_list.append(unit[6])
        P_list.append(unit[7])
    return   unit_xrd,unit_y,unit_name,anion_list,ligand_list,M_list,gas_Smile_list,P_list
    
class Encoder3D(nn.Module):
    def __init__(self, encoder,  generator,min_max_pressure):
        super(Encoder3D, self).__init__()
        self.encoder = encoder
        
        args_KANO = parse_train_args()
        modify_train_args(args_KANO)
        args_KANO.cuda=True
        args_KANO.checkpoint_path="./save/original_CMPN.pkl"
        self.gas_embed = get_cmpn_encoder(args_KANO)   
        encoder_add_functional_prompt(self.gas_embed, args_KANO)
        self.gas_embed.load_state_dict(torch.load(args_KANO.checkpoint_path, map_location='cpu'), strict=False)
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        
        self.M_embed = MetalModel() 
        self.P_embed = PressureModel(min_max_pressure)  

        self.generator = generator

    def forward_once(self, xrd_input,anion_input,ligand_input,M_input,gas_input,P_input):
        xrd_encoder_ed = self.encoder(xrd_input)

        gas_embed_ed = self.gas_embed( 'finetune', False, gas_list, None)
        anion_embed_ed = self.gas_embed( 'finetune', False, anion_input, None)
        ligand_embed_ed = self.gas_embed( 'finetune', False, ligand_input, None)
        M_embed_ed = self.M_embed(M_input)
        P_embed_ed = self.P_embed(P_input)
        all_encoder = torch.cat([xrd_encoder_ed, ligand_embed_ed], dim=-1)
        all_encoder = torch.cat([all_encoder,anion_embed_ed ], dim=-1)
        all_encoder = torch.cat([all_encoder, gas_embed_ed], dim=-1)
        all_encoder = torch.cat([all_encoder, M_embed_ed], dim=-1)
        all_encoder = torch.cat([all_encoder, P_embed_ed], dim=-1)
        
        return all_encoder

    def forward(self, xrd_input,anion_input,ligand_input,M_input,gas_input,P_input):
        output_info=self.generator(self.forward_once(xrd_input,anion_input,ligand_input,M_input,gas_input,P_input))
        return output_info
        
#AD data
gas_dict=torch.load('data/Smiles_ads.pt')
with open('data/PXRD_DICT.json', 'r') as json_file:
    DICT_XRD=json.load( json_file)
    

exAPMOF_DICT=torch.load('data/exAPMOF_DICT.pt')
A_dict=exAPMOF_DICT[0]
L_dict=exAPMOF_DICT[1]
M_dict=exAPMOF_DICT[2]
L_Reflection=exAPMOF_DICT[3]
exAPMOF_data=torch.load('data/exAPMOF_ISOdata.pt')
target_dict = ['Gas uptake']
target_dict_selected = []
tgt = [0] 
for i in range(len(tgt)):
    target_dict_selected.append(target_dict[tgt[i]])
print(target_dict_selected)

all_exAPMOF=torch.load('data/all_exAPMOF-iso.pt')
a=np.arange(len(all_exAPMOF))

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15
seednum=9764
np.random.seed(seednum)
np.random.shuffle(a)
train_idx=a[:int(len(a)*train_ratio)]
val_idx=a[int(len(a)*train_ratio):-int(len(a)*test_ratio)]
test_idx=a[-int(len(a)*test_ratio):]

set_train_MofGas_PAIR = [all_exAPMOF[i] for i in train_idx]
set_val_MofGas_PAIR = [all_exAPMOF[i] for i in val_idx]
set_test_MofGas_PAIR = [all_exAPMOF[i] for i in test_idx]
set_train,set_val,set_test=[],[],[]
for key in set_train_MofGas_PAIR:
    for item in exAPMOF_data[key]:
        mof,gas=key.split('_')
        gas_ad=item[1]
        info_ad=[mof,gas]+list(item)
        if item[0]!=0:
            set_train.append(info_ad)

for key in set_test_MofGas_PAIR:
    for item in exAPMOF_data[key]:
        mof,gas=key.split('_')
        gas_ad=item[1]
        info_ad=[mof,gas]+list(item)
        if item[0]!=0:
            set_test.append(info_ad)

for key in set_val_MofGas_PAIR:
    for item in exAPMOF_data[key]:
        mof,gas=key.split('_')
        gas_ad=item[1]
        info_ad=[mof,gas]+list(item)
        if item[0]!=0:
            set_val.append(info_ad)
data_y_train=[]
list_pressure=[]
list_gas=[]
for InputInfo in set_train:
    key=InputInfo[0]
    Adsorption_Va=[InputInfo[-1]]
    list_gas.append(InputInfo[1])
    data_y_sg=Adsorption_Va
    data_y_train.append(data_y_sg)
    pressure=math.log10(float(InputInfo[2]))
    list_pressure.append(pressure)

set_P=set(list_pressure)
set_gas=set(list_gas)
minimum_list_P = min(set_P)-1
maximum_list_P = max(set_P)+1
if minimum_list_P==0:
    minimum_list_P=1e-10
print("Minimum pressure value in the input:", minimum_list_P)
print("Maximum pressure value in the input:", maximum_list_P)
min_max_pressure=[minimum_list_P,maximum_list_P]

print(min_max_pressure,set_gas)
data_y_train=process_dataY(data_y_train)
data_y_train=data_y_train
scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(1)]

print(scales)

#model
epochs = 400
print("epoch\t loss\t")
t1 = time()
batch_size_train=128
loss_func = nn.MSELoss()
multiple_train_steps = 0.1
multiple_weight_decay = 0.001

train_loader = DataLoader(set_train, batch_size=batch_size_train, shuffle=True)
dev_loader = DataLoader(set_val, batch_size=100)
test_loader = DataLoader(set_test, batch_size=100)


train_size=int(len(set_train)/batch_size_train)
total_steps=int(train_size * epochs)
print(total_steps,train_size,len(set_train))
device = torch.device("cuda")

loss_func = nn.MSELoss()
model=build_model(tgt=tgt,min_max_pressure=min_max_pressure, embed_dim=10752,dropout=0.25).cuda()
model_path =  'save/CMPN_pre-L1.pt'
checkpoint = torch.load(model_path)
for a_name in checkpoint['model'].keys():
    a_param=checkpoint['model'][a_name]
    if 'gas_embed'  in a_name:
        b_name = a_name  
        b_param = model.state_dict()[b_name]  
        b_param.copy_(a_param.data)  
num_train_optimization_steps = int(train_size * epochs)
optim = torch.optim.AdamW(model.parameters(), lr=10.0e-5, weight_decay=multiple_weight_decay)
lr_scheduler =  transformers.get_linear_schedule_with_warmup(optim,int(num_train_optimization_steps *multiple_train_steps),num_train_optimization_steps)
bestmse=1e30
#run
for epoch in range(epochs):
    loss = 0.0
    model.train()
    train_label=[]
    train_pred=[]

    for item in train_loader:
        sub_list=tran_list(item)
        instances=[]
        for InputInfo in sub_list:
            xrd, label,A_item,L_item,M_item,gas_Smile,P_item=process_InputInfo(InputInfo)
            instances.append([xrd,label,InputInfo[0],A_item,L_item,M_item,gas_Smile,P_item])
        batch=collate_fn(instances)
        data_xrd=batch[0]
        my_array = np.array(data_xrd)
        data_xrd = torch.tensor(my_array,dtype=torch.float32)
        data_xrd = data_xrd.to(device)
        unit_y=batch[1]
        unit_y=process_dataY(unit_y)
        unit_y=unit_y[:, tgt]
        for i in range(unit_y.shape[-1]):
            unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
        labels=unit_y.to(device)
        anion_list=batch[3]
        ligand_list=batch[4]
        
        M_list=torch.tensor(batch[5])
        M_list=M_list.to(device)
        
        gas_list=batch[6]
        
        P_list=torch.tensor(batch[7])
        P_list=P_list.to(device)
        y_hat = model(data_xrd,anion_list,ligand_list,M_list, gas_list,P_list)
        train_label.append(labels)
        train_pred.append(y_hat)
        loss_batch = loss_func(y_hat,labels)
        loss = loss+loss_batch.item()
        optim.zero_grad()
        loss_batch.backward()
        optim.step()
        lr_scheduler.step()
        
        
        
    if((epoch+1)%5 == 0):
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
        
        best_model.eval()
        test_label=[]
        test_pred=[]
        mse=0
        with torch.no_grad():
            for item in dev_loader: 
                sub_list=tran_list(item)
                instances=[]
                for InputInfo in sub_list:
                    xrd, label,A_item,L_item,M_item,gas_Smile,P_item=process_InputInfo(InputInfo)
                    instances.append([xrd,label,InputInfo[0],A_item,L_item,M_item,gas_Smile,P_item])  
                batch=collate_fn(instances)
                data_xrd=batch[0]
                my_array = np.array(data_xrd)
                data_xrd = torch.tensor(my_array,dtype=torch.float32)
                data_xrd = data_xrd.to(device)
                unit_y=batch[1]
                unit_y=process_dataY(unit_y)
                unit_y=unit_y[:, tgt]
                for i in range(unit_y.shape[-1]):
                    unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
                labels=unit_y.to(device)
                anion_list=batch[3]
                ligand_list=batch[4]
                
                M_list=torch.tensor(batch[5])
                M_list=M_list.to(device)
                
                gas_list=batch[6]

                P_list=torch.tensor(batch[7])
                P_list=P_list.to(device)
                y_hat = best_model(data_xrd,anion_list,ligand_list,M_list, gas_list,P_list)
                mse += loss_func(y_hat, labels).item()
                test_label.append(labels)
                test_pred.append(y_hat)
                
        
        test_label = torch.cat(test_label).cuda()
        test_pred = torch.cat(test_pred).cuda()
        for i in range(0,len(tgt),1):
            test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
            test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
            r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
            print(f'dev {target_dict[tgt[i]]} R2: {r2}')
        print("{}\t dev mse:{:.3f} r2 {:.5f}".format(i+1,mse,r2.item()))

    mse=0
    model.eval()
    test_label=[]
    test_pred=[]
    with torch.no_grad():
        for item in dev_loader: 
                sub_list=tran_list(item)
                instances=[]
                for InputInfo in sub_list:
                    xrd, label,A_item,L_item,M_item,gas_Smile,P_item=process_InputInfo(InputInfo)
                    instances.append([xrd,label,InputInfo[0],A_item,L_item,M_item,gas_Smile,P_item])  
                batch=collate_fn(instances)
                data_xrd=batch[0]
                my_array = np.array(data_xrd)
                data_xrd = torch.tensor(my_array,dtype=torch.float32)
                data_xrd = data_xrd.to(device)
                unit_y=batch[1]
                unit_y=process_dataY(unit_y)
                unit_y=unit_y[:, tgt]
                for i in range(unit_y.shape[-1]):
                    unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
                labels=unit_y.to(device)
                anion_list=batch[3]
                ligand_list=batch[4]
                
                M_list=torch.tensor(batch[5])
                M_list=M_list.to(device)
                
                gas_list=batch[6]

                P_list=torch.tensor(batch[7])
                P_list=P_list.to(device)
                y_hat = model(data_xrd,anion_list,ligand_list,M_list, gas_list,P_list)
                mse += loss_func(y_hat, labels).item()
                test_label.append(labels)
                test_pred.append(y_hat)
            
        
    if mse < bestmse:
        best_model=copy.deepcopy(model)
        bestmse=mse
        print('new mse bestmse',epoch,bestmse,optim.param_groups[0]['lr'] * 1e5)
        
        try:
            train_label = torch.cat(train_label).cuda()
            train_pred = torch.cat(train_pred).cuda()
            for i in range(0,len(tgt),1):
                train_pred[:, i] = train_pred[:, i] * scales[i][1] + scales[i][0]
                train_label[:, i] = train_label[:, i] * scales[i][1] + scales[i][0]
                r2 = round(r2_score(train_label[:, i].detach().cpu().numpy(), train_pred[:, i].detach().cpu().numpy()), 3)
                print(f'train {target_dict[tgt[i]]} R2: {r2}')
            print("time: {:.5f}".format(time() - t1))
        except:
            for i in range(0,len(tgt),1):
                r2 = round(r2_score(train_label[:, i].detach().cpu().numpy(), train_pred[:, i].detach().cpu().numpy()), 3)
                print(f'train {target_dict[tgt[i]]} R2: {r2}')
            print("time: {:.5f}".format(time() - t1))
        print(epoch,optim.param_groups[0]['lr'] * 1e5)
        #t1 = time()
        
        test_label = torch.cat(test_label).cuda()
        test_pred = torch.cat(test_pred).cuda()
        for i in range(0,len(tgt),1):
            test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
            test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
            r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
            print(f'dev {target_dict[tgt[i]]} R2: {r2}')
        print("{}\t dev mse:{:.3f} r2 {:.5f}".format(i+1,mse,r2.item()))


best_model.eval()
test_label=[]
test_pred=[]
mse=0
with torch.no_grad():
    for item in test_loader: 
        sub_list=tran_list(item)
        instances=[]
        for InputInfo in sub_list:
            xrd, label,A_item,L_item,M_item,gas_Smile,P_item=process_InputInfo(InputInfo)
            instances.append([xrd,label,InputInfo[0],A_item,L_item,M_item,gas_Smile,P_item])  
        batch=collate_fn(instances)
        data_xrd=batch[0]
        my_array = np.array(data_xrd)
        data_xrd = torch.tensor(my_array,dtype=torch.float32)
        data_xrd = data_xrd.to(device)
        unit_y=batch[1]
        unit_y=process_dataY(unit_y)
        unit_y=unit_y[:, tgt]
        for i in range(unit_y.shape[-1]):
            unit_y[:, i] = (unit_y[:, i] - scales[i][0]) / scales[i][1]
        labels=unit_y.to(device)
        anion_list=batch[3]
        ligand_list=batch[4]

        M_list=torch.tensor(batch[5])
        M_list=M_list.to(device)

        gas_list=batch[6]

        P_list=torch.tensor(batch[7])
        P_list=P_list.to(device)
        y_hat = best_model(data_xrd,anion_list,ligand_list,M_list, gas_list,P_list)
        mse += loss_func(y_hat, labels).item()
        test_label.append(labels)
        test_pred.append(y_hat)
        

test_label = torch.cat(test_label).cuda()
test_pred = torch.cat(test_pred).cuda()
for i in range(0,len(tgt),1):
    test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
    test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
    r2 = round(r2_score(test_label[:, i].detach().cpu().numpy(), test_pred[:, i].detach().cpu().numpy()), 3)
    print(f'TEST {target_dict[tgt[i]]} R2: {r2}')
print("{}\t TEST mse:{:.3f} r2 {:.5f}".format(i+1,mse,r2.item()))

torch.save(best_model.state_dict(),'APMOF-ISO-ALM+PXRD5.pt')