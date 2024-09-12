import gc
import os
import re,csv

import transformers
import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss, relu
from sklearn.metrics import r2_score,mean_squared_error,median_absolute_error
from copy import deepcopy
import sys

sys.path.append("..")
from script_for_models.utils_dist import parse_args, Logger, set_seed
args = parse_args()
from script_for_models.model_pre import build_model_gas_only
import warnings
import time
import numpy as np

torch.cuda.set_device(0)
device = torch.device("cuda")


def process_dataY(data_y_list):
    data_y_array = np.array(data_y_list)
    newy=[]
    for i1 in data_y_array:
        a_float_m = map(float, i1)
        a_float_m = list(a_float_m)
        newy.append(a_float_m)
    data_y_tensor=torch.tensor(newy)
    return data_y_tensor


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, data_x,data_y):
    
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):

        cry = self.data_x[index]
        lab = self.data_y[index,:]
        
        return cry,lab 

    def __len__(self):
    
        return len(self.data_x)
    
    
def process_csv_file(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            #print(row)
            smiles = row['smiles']
            newin=[smiles]
            for key in row.keys():
                if key != 'smiles':
                    newin.append(row[key])
            result.append(newin)
    return result
if __name__ == "__main__":

    save_path='save/'
    log = Logger(save_path + 'Logger/', f'msa_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.log')
    set_seed(args.seed)
    args.dist_bar = [[3, 5, 8, 1e10]]
    args.fnn_dim=2048
    args.batch_size=256
    args.dropout=0.15
    args.embed_dim=512
    args.gpu_id='0'
    args.atom_class=100

    target_dict = ['TB,K','TC,K','PC,bar','VC,ml/mol','rhoC,g/ml','g/mol','omega']
    target_dict_selected = []
    tgt = [0,1,2,3,4,5,6]
    for i in range(len(tgt)):
        target_dict_selected.append(target_dict[i])
    print(target_dict_selected)
    tgt_select = [target_dict[i] for i in tgt]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    CMPN_pre_input=torch.load('data/CMPN-pre.pt')
    a=np.arange(len(CMPN_pre_input))
    train_ratio=0.95
    val_ratio=0.025
    test_ratio=0.025
    seednum=9845
    np.random.seed(seednum)
    np.random.shuffle(a)
    train_idx=a[:int(len(a)*train_ratio)]
    val_idx=a[int(len(a)*train_ratio):-int(len(a)*test_ratio)]
    test_idx=a[-int(len(a)*test_ratio):]
    print(a[1:10],len(CMPN_pre_input))
    set_train = [CMPN_pre_input[i] for i in train_idx]
    set_val = [CMPN_pre_input[i] for i in val_idx]
    set_test = [CMPN_pre_input[i] for i in test_idx]
    
    
    data_x_train = [CMPN_pre_input[i][0] for i in train_idx]
    data_x_val = [CMPN_pre_input[i][0] for i in val_idx]
    data_x_test = [CMPN_pre_input[i][0] for i in test_idx]
    
    data_y_train = [CMPN_pre_input[i][1:] for i in train_idx]
    data_y_val = [CMPN_pre_input[i][1:] for i in val_idx]
    data_y_test = [CMPN_pre_input[i][1:] for i in test_idx]
    
    data_y_all = [CMPN_pre_input[i][1:] for i in a]
    
    data_y_train=process_dataY(data_y_train)
    data_y_val=process_dataY(data_y_val)
    data_y_test=process_dataY(data_y_test)
    
    data_y_all=process_dataY(data_y_all)
    
    data_y_train=data_y_train[:, tgt]
    data_y_val=data_y_val[:, tgt]
    data_y_test=data_y_test[:, tgt]
    
    data_y_all=data_y_all[:, tgt]
    
    train_dataset=SmilesDataset(data_x_train, data_y_train)
    dev_dataset=SmilesDataset(data_x_val, data_y_val)
    test_dataset=SmilesDataset(data_x_test, data_y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32)
    test_dataset = DataLoader(test_dataset, batch_size=32)
    
    train_size=len(data_x_train)
    val_size=len(data_x_val)
    test_size=len(data_x_test)
    
    scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(data_y_train.shape[-1])]
    
    print(scales)
    
    print(data_y_all.shape,scales)
    for i in range(data_y_train.shape[-1]):
        data_y_train[:, i] = (data_y_train[:, i] - scales[i][0]) / scales[i][1]
    for i in range(data_y_test.shape[-1]):
        data_y_test[:, i] = (data_y_test[:, i] - scales[i][0]) / scales[i][1]    
    for i in range(data_y_val.shape[-1]):
        data_y_val[:, i] = (data_y_val[:, i] - scales[i][0]) / scales[i][1]
    
    del CMPN_pre_input,data_y_all,data_x_train, data_y_train,data_x_val, data_y_val,data_x_test, data_y_test
    
    criterion = torch.nn.MSELoss()
    best_loss, best_mse = 1e18, 1e18
    early_stop = 0
    epochnumber=200
    epoch = 0


    model=build_model_gas_only(tgt=len(tgt), ffn_dim=args.fnn_dim,dropout=args.dropout).cuda()
    print('model_build_successful!')
    num_train_optimization_steps = int(len(train_loader) * epochnumber)
    print(num_train_optimization_steps)
    multiple_train_steps = 0.1
    multiple_weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=10.0e-5 * len(args.gpu_id.split(',')), weight_decay=multiple_weight_decay)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer,int(num_train_optimization_steps *multiple_train_steps),num_train_optimization_steps)
    
    t0 = time.time()
    
    for epoch in range(0,epochnumber):
        model.train()
        loss = 0.0
        t1 = time.time()

        train_pred=[]
        train_label=[]
        
        for gas_list, labels in train_loader:
            labels=labels.cuda()
            pred = model(gas_list)
            train_pred.append(pred.clone().detach())
            train_label.append(labels.clone().detach())
            loss_batch = criterion(pred, labels)
            loss += loss_batch.item()
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            lr_scheduler.step()
            
        train_pred = torch.cat(train_pred)
        train_label = torch.cat(train_label)
        
        for i in range(0,len(tgt),1):
            train_pred[:, i] = relu(train_pred[:, i] * scales[i][1] + scales[i][0])
            train_label[:, i] = train_label[:, i] * scales[i][1] + scales[i][0]
            
            pred_for_loss=train_pred[:, i].detach().cpu().numpy()
            labels_for_loss=train_label[:, i].detach().cpu().numpy()
            mse = round(mean_squared_error(pred_for_loss, labels_for_loss), 3)
            rmse = round(np.sqrt(mse)   , 3)
            mae = round(median_absolute_error(pred_for_loss, labels_for_loss), 3)
            r2 = round(r2_score(labels_for_loss, pred_for_loss), 3)
            log.logger.info(f'train {tgt_select[i]} R2: {r2} mse: {mse} rmse: {rmse} mae: {mae}')
        
        TOTALmse = torch.nn.functional.mse_loss(train_pred, train_label)
        TOTALrmse = torch.sqrt(TOTALmse)
        TOTALmae  = torch.nn.functional.l1_loss(train_pred, train_label)
        log.logger.info(f'TOTAL train  mse: {TOTALmse} rmse: {TOTALrmse} mae: {TOTALmae}')
        
        torch.cuda.empty_cache()
        
        model.eval()
        mse = 0
        dev_pred=[]
        dev_label=[]
        for gas_list, labels  in dev_loader:
        
            labels=labels.cuda()
           
            with torch.no_grad(): 
                pred = model(gas_list)
                mse += mse_loss(pred[:, -1], labels[:, -1], reduction='sum').item() / test_size * scales[-1][1]
                dev_pred.append(pred)
                dev_label.append(labels)
        dev_pred = torch.cat(dev_pred)
        dev_label = torch.cat(dev_label)
        for i in range(0,len(tgt),1):
            dev_pred[:, i] = relu(dev_pred[:, i] * scales[i][1] + scales[i][0])
            dev_label[:, i] = dev_label[:, i] * scales[i][1] + scales[i][0]
            
            pred_for_loss=dev_pred[:, i].cpu().numpy()
            labels_for_loss=dev_label[:, i].cpu().numpy()
            mse = round(mean_squared_error(pred_for_loss, labels_for_loss), 3)
            rmse = round(np.sqrt(mse)   , 3)
            mae = round(median_absolute_error(pred_for_loss, labels_for_loss), 3)
            r2 = round(r2_score(labels_for_loss, pred_for_loss), 3)
            log.logger.info(f'Dev {tgt_select[i]} R2: {r2} mse: {mse} rmse: {rmse} mae: {mae}')
            
        TOTALmse = torch.nn.functional.mse_loss(dev_pred, dev_label)
        TOTALrmse = torch.sqrt(TOTALmse)
        TOTALmae  = torch.nn.functional.l1_loss(dev_pred, dev_label)
        log.logger.info(f'TOTAL Dev   mse: {TOTALmse} rmse: {TOTALrmse} mae: {TOTALmae}')
        
        if mse < best_mse:
            best_mse = deepcopy(mse)
            best_model = deepcopy(model)
            best_epoch = epoch + 1
            checkpoint = {'model': best_model.state_dict(), 'n_tgt': len(tgt), 'epochs': best_epoch, 'lr': optimizer.param_groups[0]['lr'],'scales':scales}
            if len(args.gpu_id) > 1: checkpoint['model'] = best_model.module.state_dict()
            torch.save(checkpoint, save_path + 'CMPN_pre.pt')
        if loss < best_loss:
            best_loss = deepcopy(loss)
            early_stop = 0
        else:
            early_stop += 1
        log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MSE: {:.2f} | Lr: {:.1f}'.
                        format(epoch + 1, time.time() - t1, loss, mse, optimizer.param_groups[0]['lr'] * 1e5))
        if early_stop >= 40:
            log.logger.info(f'Early Stopping!!! No Improvement on Loss for 20 Epochs.')
            break
            
        test_pred, test_label = [], []
        best_model.eval()
            
        for gas_list, labels in test_dataset:
            labels=labels.cuda()
            with torch.no_grad():
                pred = best_model(gas_list)
                test_pred.append(pred)
                test_label.append(labels)
        test_pred = torch.cat(test_pred)
        test_label = torch.cat(test_label)
        for i in range(0,len(tgt),1):
            test_pred[:, i] = relu(test_pred[:, i] * scales[i][1] + scales[i][0])
            test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
            pred_for_loss=test_pred[:, i].cpu().numpy()
            labels_for_loss=test_label[:, i].cpu().numpy()
            mse = round(mean_squared_error(pred_for_loss, labels_for_loss), 3)
            rmse = round(np.sqrt(mse) , 3)
            mae = round(median_absolute_error(pred_for_loss, labels_for_loss), 3)
            r2 = round(r2_score(labels_for_loss, pred_for_loss), 3)
            
            
            log.logger.info(f'Test {tgt_select[i]} R2: {r2} mse: {mse} rmse: {rmse} mae: {mae}')
    
        TOTALmse = torch.nn.functional.mse_loss(test_pred, test_label)
        TOTALrmse = torch.sqrt(TOTALmse)
        TOTALmae  = torch.nn.functional.l1_loss(test_pred, test_label)
        log.logger.info(f'TOTAL Test :  mse: {TOTALmse} rmse: {TOTALrmse} mae: {TOTALmae}')
        
        
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 40, (time.time() - t0) / 3600, "=" * 40))
    log.logger.info('Dist_bar: {} | Best Epoch: {} | Dev_MSE: {:.2f}'.format(args.dist_bar, best_epoch, best_mse))
    checkpoint = {'model': best_model.state_dict(), 'n_tgt': len(tgt), 'dist_bar': args.dist_bar,
                    'epochs': best_epoch, 'lr': optimizer.param_groups[0]['lr'],'scales':scales}
    if len(args.gpu_id) > 1: checkpoint['model'] = best_model.module.state_dict()
    torch.save(checkpoint, save_path + 'CMPN_pre-L1.pt')
    log.logger.info('Save the best model as CMPN_pre-L1.pt')
