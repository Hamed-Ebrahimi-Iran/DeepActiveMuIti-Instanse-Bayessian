import torch
import torch.nn as nn
import torch.nn.init as torch_init 


import option
#torch.set_default_tensor_type('torch.FloatTensor')
#torch.set_printoptions(profile="full")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Model(nn.Module):
    def __init__(self, n_features,len_ninput,len_ainput,batch_size):
        super(Model, self).__init__()

       # args = option.parser.parse_args()
        self.n_features=n_features
        self.len_ninput=len_ninput
        self.len_ainput=len_ainput
        self.batch_size=batch_size 
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10
        #self.Aggregate = Aggregate(len_feature=2048)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.BatchNorm1d=nn.BatchNorm1d
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self,n_features,ninput,ainput,batch_size):
        
        inputs = torch.cat((ninput, ainput), 0).to(device) 
        batch_size=self.batch_size
        k_abn = self.k_abn
        k_nor = self.k_nor
        bs, ncrops, t, f = inputs.size() #torch.Size([4, 10, 32, 2048])
        out = inputs.view(-1, t, f) #torch.Size([40, 32, 2048])
        #features = out           #torch.Size([40, 32, 2048]) 4*10,32,2048 --> 20 anomaly and 20 normal
        #out = out.permute(0, 2, 1) #torch.Size([40, 2048, 32]) --> error 2048

        #out = self.Aggregate(out)
        #out = self.drop_out(out)
        features = out
        scores = self.relu(self.fc1(features))
        scores = self.dropout(scores)
        #hidden = x
        scores = self.relu(self.fc2(scores))
        scores = self.dropout(scores)
        scores = self.relu(self.fc3(scores))#torch.Size([40, 32, 1])
        scores = self.sigmoid(scores) #torch.Size([40, 32, 1])
        #scores = self.sigmoid(self.fc3(scores)) #torch.Size([40, 32, 1])
        #x = self.dropout(x)

        scores = scores.view(bs, ncrops,-1 ).mean(1) #torch.Size([40, 32])
        scores = scores.unsqueeze(dim=2) #torch.Size([40, 32,1])

        normal_features = features[0:self.len_ninput*ncrops] # features[0:2*10]
        normal_scores = scores[0:self.len_ninput]

        abnormal_features = features[self.len_ainput*ncrops:] # features[2*10:]
        abnormal_scores = scores[self.len_ainput:]  
        
        feat_magnitudes = torch.norm(features, p=2, dim=2) #torch.Size([40, 32],dim=2) and torch.Size([40, 2048],dim=1)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.len_ninput]  # normal feature magnitudes [0:self.batchsize]
        afea_magnitudes = feat_magnitudes[self.len_ainput:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        #select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx #tensor.size([1,32])
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1] #tensor([[13, 11, 10]])
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]]) #torch.Size([1, 3, 2048])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f) #torch.Size([1, 10, 32, 2048])
        abnormal_features = abnormal_features.permute(1, 0, 2,3) #torch.Size([10, 1, 32, 2048])

        total_select_abn_feature = torch.zeros(0, device=inputs.device) #torch.Size([10, 3, 2048])
        for abnormal_feature in abnormal_features:    #torch.Size([1, 32, 2048])
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)  #torch.Size([1, 3, 2048]) # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn)) #torch.Size([1, 3, 2048])

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        #select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3) #torch.Size([10, 1, 32, 2048])

        total_select_nor_feature = torch.zeros(0, device=inputs.device) #torch.Size([10, 3, 2048])
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]]) #torch.Size([1, 3, 1])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag
        
        feat_select_abn = total_select_abn_feature  #torch.Size([10, 3, 2048])
        feat_select_normal = total_select_nor_feature #torch.Size([10, 3, 2048])
        return scores,normal_scores,score_normal,abnormal_scores,score_abnormal,feat_select_abn ,feat_select_normal



