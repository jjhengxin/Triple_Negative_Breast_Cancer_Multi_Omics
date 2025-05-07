import torch
from torch_geometric.data import Data
import pandas as pd
import random
import numpy as np



from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)


def make_data_geo(data_geo, label_geo,k,i,seed):
    assert k > 1
    
    data = Data()
    np.random.seed(seed)
    indices = np.random.permutation(range(len(label_geo)))
    # X = data_geo.loc[indices]
    # Y = label_geo.loc[indices]
    X = torch.tensor(data_geo.loc[indices].values,dtype=torch.float)
    Y = torch.tensor(label_geo.loc[indices,:].values,dtype=torch.int)

    fold_size = X.shape[0] // k
    X_train, Y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], Y[idx]
        if j == i: ###第i折作valid
            X_test, Y_test = X_part, y_part
        elif X_train is None:
            X_train, Y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            Y_train = torch.cat((Y_train, y_part), dim=0)


    data.X_train= X_train
    data.X_test= X_test
    data.Y_train= Y_train
    data.Y_test= Y_test

    # pd.DataFrame(X_test).to_csv(r'result/X_test.csv')
    # pd.DataFrame(Y_test).to_csv(r'result/Y_test.csv')
    return data


def get_train_edge(data_edge_index, train_anchor):
    train_edge_index = pd.DataFrame(dtype=int)
    test_edge_index = pd.DataFrame(dtype=int)

    for i in range(len(data_edge_index)):
        if(i%10000 == 0):
            print(i/len(data_edge_index))
        if (data_edge_index.iloc[i,0] in train_anchor.values) or (data_edge_index.iloc[i,1] in train_anchor.values):
            train_edge_index = train_edge_index._append(data_edge_index.iloc[i,:])
        # elif(data_edge_index.iloc[i,0] in train_anchor.values) or (data_edge_index.iloc[i,1] in train_anchor.values):
        #     test_edge_index = test_edge_index._append(data_edge_index.iloc[i,:])
    return train_edge_index , test_edge_index



# ,data_ATAC1_link_index,data_ATAC2_link_index
def make_data(data_x,data_ppi_link_index,data_homolog_index,data_ATAC1_link_index,data_ATAC2_link_index,anchor_list,test_anchor,seed):
    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    not_anchor_index = anchor_list.result_num[anchor_list.result_num==0].index

    train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))
    not_train_anchor = pd.Series(list(set(anchor_list.index)-set(train_anchor.to_list())))

    data_y = pd.Series(0,index=data_x.index,dtype=int)
    data_y[anchor_index.to_list()]=1

    # test_sample = random.sample(not_anchor_index.to_list(),len(anchor_index))
    random.seed(seed)
    test_sample = random.sample(not_train_anchor.to_list(),len(train_anchor))

    data_train_mask = pd.Series(False,index=data_x.index,dtype=bool)
    data_train_mask[train_anchor.to_list()]=True
    data_train_mask[test_sample]=True

    # data_test_mask = pd.Series(False,index=data_x.index,dtype=bool)
    # data_test_mask[test_anchor.to_list()]=True
    # data_test_mask[test_sample[len(train_anchor):]]=True
    data_test_mask = pd.Series(True,index=data_x.index,dtype=bool)
    data_test_mask[data_train_mask]=False
    
    



    data = Data()
    data.num_nodes = len(data_x)
    data.num_node_features = data_x.shape[1]
    data.edge_index = {
                       'ppi':torch.tensor(data_ppi_link_index.T.values,dtype=torch.long),
                       'homolog':torch.tensor(data_homolog_index.T.values,dtype=torch.long),
                       'ATAC1':torch.tensor(data_ATAC1_link_index.T.values,dtype=torch.long),
                       'ATAC2:':torch.tensor(data_ATAC2_link_index.T.values,dtype=torch.long)
                       }
    

    data.x = torch.tensor(data_x.values,dtype=torch.float)
    data.y = torch.tensor(data_y.values,dtype=torch.int)
    data.train_mask = torch.tensor(data_train_mask.values,dtype=torch.bool)
    data.test_mask = torch.tensor(data_test_mask.values,dtype=torch.bool)
    return data


