import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import numpy as np


from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)

EPSILON = np.finfo(float).eps



class Decoder(nn.Module):
    def __init__(self,feature_num,gene_num ):
        super(Decoder,self).__init__()

        
        self.decoder = nn.Sequential(
            nn.Linear(feature_num,60),
            nn.ReLU(),
            nn.Linear(60,250),
            nn.ReLU(),
            nn.Linear(250,1000),
            nn.ReLU(),
            nn.Linear(1000,5000),
            nn.ReLU(),
            nn.Linear(5000,10000),
            nn.ReLU(),
            nn.Linear(10000,gene_num),
            nn.Sigmoid()
        )
        self.loss = nn.MSELoss()
    def forward(self,x,y):
        x = self.decoder(x)
        x = (x -0.5)*2
        loss = self.loss(x,y)
        return {"out":x,"loss":loss}







class MutiGAT(nn.Module):
    def __init__(self,num_muti_gat,num_node_features, hid_c, out_c,data_x_N,num_node_features_cellChat):
        super(MutiGAT,self).__init__()
        self.num_muti_gat = num_muti_gat
        # self.num_muti_graph = num_muti_graph
        self.data_x_N = data_x_N
        self.gat_list = []
        self.edge_type_list = ["ppi","homolog","ATAC1"]
        
        # self.graph_list= []
        self.cellChat_gat = GraphAT_cellChat(in_c=num_node_features_cellChat, hid_c=30, out_c=1)
        for i in range(num_muti_gat):
            self.gat_list.append(GraphAT(in_c=num_node_features, hid_c=hid_c, out_c=out_c))
        
        
        self.gat_list = nn.ModuleList(self.gat_list)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        
    def forward(self,data,data_cellChat,edge_cellChat):
        out = torch.zeros_like(data.y)
        loss = torch.Tensor([0.]).to(next(self.parameters()).device)
        i = 0
        # graph_loss = torch.Tensor([0.]).to(next(self.parameters()).device)
        # graph = torch.zeros(self.data_x_N,self.data_x_N,dtype=torch.float32).to(next(self.parameters()).device)
        out_cc = self.cellChat_gat(data_cellChat,edge_cellChat)
        
        x = data.x * out_cc.T
        for module in self.gat_list:
            temp,temp_loss = module(x,data,self.edge_type_list[(i%2)])##########
            i=i+1
            out = out + temp[:,1].exp()
            
            loss = loss + temp_loss

        # return out/(self.num_muti_gat), loss/self.num_muti_gat
        return {'out':out/(self.num_muti_gat),'loss_mutiGAT':loss/self.num_muti_gat}
    
class GraphAT(nn.Module):
    def __init__(self ,in_c, hid_c, out_c):
        super(GraphAT, self).__init__()  
        
        
        # self.conv_cell = pyg_nn.GATConv(in_channels=in_celltype, out_channels=hid_c, dropout=0.6 ,heads=1, concat=False)  # [N, N], N为celltype的数量
        self.conv1 = pyg_nn.GATConv(in_channels=in_c, out_channels=hid_c, dropout=0.6 ,heads=3, concat=False)
        self.bn1   = nn.BatchNorm1d(hid_c)
        self.conv2 = pyg_nn.GATConv(in_channels=hid_c, out_channels=out_c, dropout=0.6, heads=3, concat=False)
        

    def forward(self, x,data,edge_type):
        # data.x  data.edge_index
        # x = data.x  # [N, C], C为特征的维度
        # x = self.dfr(x)
        edge_index = data.edge_index[edge_type]  # [2, E], E为边的数量
        x = F.dropout(x, p=0.6, training=self.training)
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D], N是节点数量，D是第一层输出的隐藏层的维度
        hid = self.bn1(hid)
        hid = F.leaky_relu(hid)
        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c], out_c就是定义的输出
        out = F.log_softmax(out, dim=1)  # [N, out_c],表示输出
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask].long())
        return out,loss
    
class GraphAT_cellChat(nn.Module):
    def __init__(self ,in_c, hid_c, out_c):
        super(GraphAT_cellChat, self).__init__()  
        
        
        # self.conv_cell = pyg_nn.GATConv(in_channels=in_celltype, out_channels=hid_c, dropout=0.6 ,heads=1, concat=False)  # [N, N], N为celltype的数量
        self.conv1 = pyg_nn.GATConv(in_channels=in_c, out_channels=hid_c, dropout=0.6 ,heads=1, concat=False)
        self.bn1   = nn.BatchNorm1d(hid_c)
        self.conv2 = pyg_nn.GATConv(in_channels=hid_c, out_channels=out_c, dropout=0.6, heads=1, concat=False)
        

    def forward(self, data,edge_index):
        # data.x  data.edge_index
        x = data  # [N, C], C为特征的维度
        # x = self.dfr(x)
        # edge_index [2, E], E为边的数量
        x = F.dropout(x, p=0.6, training=self.training)
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D], N是节点数量，D是第一层输出的隐藏层的维度
        hid = self.bn1(hid)
        hid = F.leaky_relu(hid)
        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c], out_c就是定义的输出
        out = F.softmax(out, dim=1)  # [N, out_c],表示输出
        return out
    

class ConcreteDropout(nn.Module):
    def __init__(self,shape,temp= 1.0/10.0):
        super().__init__()
        self.logit_p = nn.Parameter(torch.zeros(shape))
        self.temp = temp
    def forward(self,x):
        if self.training:
            unif_noise = torch.rand_like(self.logit_p)
            # unif_noise = torch.full_like(self.logit_p, 0.5)
        else:
            unif_noise = torch.full_like(self.logit_p, 0.5)
        dropout_p = torch.sigmoid(self.logit_p)
        approx = (
            torch.log(dropout_p + EPSILON)
            - torch.log(1. - dropout_p + EPSILON)
            + torch.log(unif_noise + EPSILON)
            - torch.log(1. - unif_noise + EPSILON)
        )
        approx_output = torch.sigmoid(approx / self.temp)


        return x*(1 - approx_output), (1-dropout_p)
    

class VariableDropoutMLP(nn.Module):
    def __init__(self,data_x_shape,temp= 1.0/10.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_x_shape,3000),  #3000
            nn.BatchNorm1d(3000),
            nn.Dropout(0.5),#0.9
            nn.LeakyReLU(),
            nn.Linear(3000,1000),#1000
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),#0.9
            nn.LeakyReLU(),
            nn.Linear(1000,500),#300
            nn.BatchNorm1d(500),
            nn.Dropout(0.5),#0.8
            nn.LeakyReLU(),
            nn.Linear(500,300),#300
            nn.BatchNorm1d(300),
            nn.Dropout(0.5),#0.8
            nn.LeakyReLU(),
            nn.Linear(300,20),
            nn.BatchNorm1d(20),
            
            nn.Linear(20,1),
            
        )
        self.logit_p = nn.Parameter(torch.zeros(data_x_shape))
        self.temp = temp
    def LogLikelihood(self,logits, fail_indicator):
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = torch.cumsum(logits, 0)
        hazard_ratio = torch.exp(logits)
        cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)

        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)

        # negative average log-likelihood
        observations = torch.sum(fail_indicator, 0)
        return 1.0*logL / observations
    
    
    def forward(self,x,fail_indicator):
        if self.training:
            unif_noise = torch.rand_like(self.logit_p)
            # unif_noise = torch.full_like(vimp, 0.5)
        else:
            unif_noise = torch.full_like(self.logit_p, 0.5)
        dropout_p = torch.sigmoid(self.logit_p)
        approx = (
            torch.log(dropout_p + EPSILON)
            - torch.log(1-dropout_p + EPSILON)
            + torch.log(unif_noise + EPSILON)
            - torch.log(1. - unif_noise + EPSILON)
        )
        approx_output = torch.sigmoid(approx / self.temp)

        out = self.model(x*torch.ones_like(approx_output))
        out = self.model(x*(1- approx_output))
        out = torch.sigmoid(out)
        nall = self.LogLikelihood(out, fail_indicator)
        
        ##
        l2 = torch.mean(torch.pow(1-approx_output,2))

        return out,(1-dropout_p),nall,l2




    


class Model(nn.Module):
    def __init__(self,data_geo_x_shape,num_muti_mlp):
        super(Model, self).__init__()
        self.num_muti_mlp = num_muti_mlp
        self.vdMLP_list = []
        for i in range(num_muti_mlp):
            self.vdMLP_list.append(VariableDropoutMLP(data_x_shape=data_geo_x_shape[1]))
        self.vdMLP_list = nn.ModuleList(self.vdMLP_list)
    def forward(self,data_geo_x,fail_indicator):
        out = torch.zeros([data_geo_x.shape[0],1]).to(next(self.parameters()).device)
        # vimp = torch.zeros([data_geo_x.shape[0]]).to(next(self.parameters()).device)
        nall = torch.Tensor([0.]).to(next(self.parameters()).device)
        l2 = torch.Tensor([0.]).to(next(self.parameters()).device)
        for module in self.vdMLP_list:
            out_t,vimp_t,nall_t,l2_t = module(data_geo_x,fail_indicator)
            out = out + out_t
            nall = nall + nall_t
            l2 = l2 + l2_t
            # vimp = vimp + vimp_t
        
        return {'out':out/self.num_muti_mlp,'loss_nall':nall/self.num_muti_mlp,'loss_L2':l2/self.num_muti_mlp}#,'vimp':vimp}