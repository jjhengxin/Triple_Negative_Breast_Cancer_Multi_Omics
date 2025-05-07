import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score,confusion_matrix,accuracy_score
from sklearn import model_selection
import numpy as np
from scipy.special import erfinv 
from lifelines.utils import concordance_index

import dataLoader as dl
import Model as md

from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed_np = 4709


EPSILON = np.finfo(float).eps

#111
data_exp = pd.read_csv(r"./data/dataset/TCGA-BRCA_rank_MRI.csv", header= 0).iloc[:,2:]
mri_f = pd.read_csv(r"./data/dataset/MRI_Features.csv",header=0).iloc[:,2:]


rankGauss = ((data_exp.values-data_exp.values.min(axis = 1,keepdims = True))/(data_exp.values.max(axis = 1,keepdims = True)-data_exp.values.min(axis = 1,keepdims = True))-0.5)*2
rankGauss = np.clip(rankGauss, -1+EPSILON, 1-EPSILON)
rankGauss = erfinv(rankGauss) 
data_exp = pd.DataFrame(rankGauss,columns=data_exp.columns)


data_exp_obj = torch.tensor(data_exp.values,dtype=torch.float)
mri_f_obj = torch.tensor(mri_f.values,dtype=torch.float)


my_net = md.Decoder(feature_num = mri_f.shape[1],gene_num = data_exp.shape[1])




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 检查设备
my_net = my_net.to(device)  
    
data_exp_obj = data_exp_obj.to(device)
mri_f_obj = mri_f_obj.to(device)

optimizer = torch.optim.Adam(my_net.parameters(), lr=0.005)  # 优化器

my_net.train()

for epoch in range(1000):
    optimizer.zero_grad()
    result = my_net(mri_f_obj,data_exp_obj)
    out=result['out']
    loss_MSE=result['loss']
    loss = loss_MSE
    loss.backward()
    optimizer.step()
    print("epoch:{},loss:{}".format(epoch + 1,loss.item()))

print("done")
my_net.eval()
torch.save(my_net,"result/model/model_AE_MRI.pt")
