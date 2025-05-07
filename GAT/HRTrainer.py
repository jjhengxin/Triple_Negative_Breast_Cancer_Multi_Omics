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





seed_np = 4709


EPSILON = np.finfo(float).eps

#111
data_geo = pd.read_csv(r"C:/Users/User/Desktop/demo/data/dataset/TCGA-BRCA_rank_vimp.csv", header= 0).iloc[:,1:]
label_geo = pd.read_csv(r"C:/Users/User/Desktop/demo/data/dataset/clinical_filtered.csv",header=0).iloc[:,1:]


########################C:/Users/User/Desktop/demo/data


rankGauss = ((data_geo.values-data_geo.values.min(axis = 1,keepdims = True))/(data_geo.values.max(axis = 1,keepdims = True)-data_geo.values.min(axis = 1,keepdims = True))-0.5)*2
rankGauss = np.clip(rankGauss, -1+EPSILON, 1-EPSILON)
rankGauss = erfinv(rankGauss) 
data_geo = pd.DataFrame(rankGauss,columns=data_geo.columns)





def test(model,data_geo):
    model.eval()
    result_train= model(data_geo.X_train,data_geo.Y_train[:,0])
    # out,prediction,_ ,_,_,temp= model(data,data_geo.X_test)
    result= model(data_geo.X_test,data_geo.Y_test[:,0])
    out = result['out']
    out_train = result_train['out']
    ci = calc_concordance_index(out,data_geo.Y_test[:,0],data_geo.Y_test[:,1])
    ci_train = calc_concordance_index(out_train,data_geo.Y_train[:,0],data_geo.Y_train[:,1])

    
    

    model.train()
    
    return {'ci':ci,'ci_train':ci_train}


def calc_concordance_index(logits, fail_indicator, fail_time):
    
    logits = logits.detach().cpu().numpy()
    fail_indicator = fail_indicator.detach().cpu().numpy()
    fail_time = fail_time.detach().cpu().numpy()

    hr_pred = -logits 
    ci = concordance_index(fail_time,
                            hr_pred,
                            fail_indicator)
    return ci



def train(k,i,seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
    
    df_acc = pd.DataFrame(columns=('epoch','ci_train','loss','ci'))
    
    data_geo_obj = dl.make_data_geo(data_geo, label_geo,k=k,i=i,seed = seed)
    

    my_net = md.Model(data_geo_x_shape=data_geo_obj.X_train.shape,num_muti_mlp=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 检查设备
    my_net = my_net.to(device)  
     
    data_geo_obj = data_geo_obj.to(device)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.005)  # 优化器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.93303)

    auc_stock = 0.0
    num = 10
    my_net.train()
    for epoch in range(1000):
        optimizer.zero_grad()

        



        
        result = my_net(data_geo_obj.X_train,data_geo_obj.Y_train[:,0])  # 预测结果
        out=result['out']
        loss_nall=result['loss_nall']
        loss_L2=result['loss_L2']
        
        loss =   1.0*loss_nall +1.0*loss_L2


       


        loss.backward()
        optimizer.step()  # 优化器
        # scheduler.step()
        test_= test(my_net, data_geo_obj)
        print("i:{},epoch:{},ci_train:{},loss:{},loss_nall:{},loss_L2:{},ci:{}".format(i,epoch + 1,test_['ci_train'] ,loss.item(),loss_nall.item(),loss_L2.item(),test_['ci']))
        df_acc=df_acc._append(pd.DataFrame({'epoch':[epoch],'ci_train':[test_['ci_train']],'loss':[loss.item()],'ci':[test_['ci']]}),ignore_index=True)
        


        
        

    
    # model test
    
    my_net.eval()

    torch.save(my_net,"result/model/model"+str(i)+".pt")

    result = my_net(data_geo_obj.X_test, data_geo_obj.Y_test[:,0])
    
    pd.DataFrame(result['out'].detach().cpu().numpy(),columns=['predict']).to_csv("result/predict_out_k="+str(k)+"_i="+str(i)+".csv",index=False)
    
    
    df_acc.to_csv("result/lossAndAcc_k="+str(k)+"_i="+str(i)+".csv")

    test_= test(my_net, data_geo_obj)
    return test_

def main():
    k=5
    ci_geo_stock = 0
    
    for i in range(k):
        test_ = train(k,i,seed_np)
        ci_geo_stock = ci_geo_stock + test_['ci']
        
    
    print( "ci:{}".format(ci_geo_stock/k))


if __name__ == "__main__":
    main()
    print("finished")