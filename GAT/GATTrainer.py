import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score,confusion_matrix,accuracy_score
from sklearn import model_selection
import numpy as np
from scipy.special import erfinv 


import dataLoader as dl
import Model as md

from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)


# 
seed = 47095
seed_np = 47095

if os.path.exists('result/'):
    pass
else:
    os.mkdir('result/')
    os.mkdir('result/model/')




#
anchor_list = pd.read_csv(r"C:/Users/User/Desktop/demo/data/dataset/hub_gene_1200.csv", header= 0)


##############
anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
train_anchor,test_anchor = model_selection.train_test_split(anchor_index, test_size=0.5)
test_anchor_csv=pd.DataFrame(test_anchor,dtype=int)
test_anchor_csv.to_csv(r'result/test_anchor.csv')



#################



#

data_x = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/scRNA_CITE_celltype_isfeature.csv',header=0).iloc[:,1:]

data_ppi_link_index = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/ppi_link_600.csv',header=0)

data_homolog_index = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/homolog.csv',header=0)

data_ATAC1_link_index = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/ATAC1_GSE212707_tf_net.csv',header=0)
data_ATAC2_link_index = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/ATAC2_tf_net.csv',header=0)

cellChat_attr = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/cellchat_attr.csv',header=0).iloc[:,1:]
cellChat_network = pd.read_csv(r'C:/Users/User/Desktop/demo/data/dataset/cellChat_network.csv',header=0)
data_cellChat_obj = torch.tensor(cellChat_attr.values,dtype=torch.float)
edge_cellChat_obj = torch.tensor(cellChat_network.T.values,dtype=torch.long)




data_obj = dl.make_data(data_x,data_ppi_link_index,data_homolog_index,data_ATAC1_link_index,data_ATAC2_link_index,anchor_list,test_anchor,seed)



def get_metrics(out_, edge_label_):
    out = out_.detach().cpu().numpy()
    edge_label = edge_label_.detach().cpu().numpy()



    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    f1 = f1_score(edge_label, pred)
    accuracy = accuracy_score(edge_label, pred)
    ap = average_precision_score(edge_label, out)

    return auc, f1, ap,accuracy

def test(model,data,data_cellChat,edge_cellChat):
    model.eval()
    target = data.y
    
    result= model(data,data_cellChat,edge_cellChat)
    
    
    out=result['out']


    
    auc,f1,ap,accuracy = get_metrics(out[data.test_mask], target[data.test_mask])
    
    
    

    model.train()
    # return auc, f1, ap, auc_geo,auc_geo_train,auc_temp,f1_geo,ap_geo
    return {'auc':auc,'f1':f1,'ap':ap,'acc':accuracy}






def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
    
    df_acc = pd.DataFrame(columns=('epoch','auc','f1','ap','loss','acc'))
    
    
    

    my_net = md.MutiGAT(num_muti_gat=9,num_node_features=data_obj.num_node_features,hid_c=20, out_c=2,data_x_N=data_obj.train_mask.shape[0],num_node_features_cellChat = cellChat_attr.shape[1])
    # my_net = GraphCNN(in_c=data_obj.num_node_features, hid_c=8, out_c=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 检查设备
    my_net = my_net.to(device)  
    data = data_obj.to(device)
    data_cellChat = data_cellChat_obj.to(device)
    edge_cellChat = edge_cellChat_obj.to(device)
    
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.005)  # 优化器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.93303)
    alpha = 0.5
    auc_stock = 0.0
    num = 10
    my_net.train()
    for epoch in range(1000):
        optimizer.zero_grad()

        



        
        result = my_net(data,data_cellChat,edge_cellChat)  # 预测结果
        out=result['out']
        loss_mutiGAT=result['loss_mutiGAT']
        
        loss =   loss_mutiGAT + 0.1 * torch.mean(torch.pow(out,2))


        


        loss.backward()
        optimizer.step()  # 优化器
        # scheduler.step()
        test_= test(my_net, data,data_cellChat,edge_cellChat)
        print("epoch:{},auc:{},loss:{}".format(epoch + 1, test_['auc'] , loss.item()))
        df_acc=df_acc._append(pd.DataFrame({'epoch':[epoch],'auc':[test_['auc']],'loss':[loss.item()]}),ignore_index=True)
        


        
        

    
    # model test
    
    my_net.eval()

    torch.save(my_net,"result/model/gat_model.pt")

    result = my_net(data,data_cellChat,edge_cellChat)
    pd.DataFrame({"predict":result['out'].detach().cpu()}).to_csv("result/predict_muti_all.csv",index=False)
    
    df_acc.to_csv("result/lossAndAcc.csv")

    test_= test(my_net, data,data_cellChat,edge_cellChat)
    return test_






if __name__ == "__main__":
    train()
    print("finished")