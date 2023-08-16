# import numpy as np
# import torch as torch
# import torch.nn.functional as F
# import option as option
# from torch.nn import L1Loss
# from torch.nn import MSELoss
# from sklearn.semi_supervised import LabelSpreading

# #torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.set_default_tensor_type('torch.FloatTensor')
# args = option.parser.parse_args()



# def sparsity(arr, batch_size, lamda2):
#     loss = torch.mean(torch.norm(arr, dim=0))
#     return lamda2*loss


# def smooth(arr, lamda1):
#     arr2 = torch.zeros_like(arr)
#     arr2[:-1] = arr[1:]
#     arr2[-1] = arr[-1]

#     loss = torch.sum((arr2-arr)**2)
#     return lamda1*loss


# def l1_penalty(var):
#     return torch.mean(torch.norm(var, dim=0))


# class SigmoidMAELoss(torch.nn.Module):
#     def __init__(self):
#         super(SigmoidMAELoss, self).__init__()
#         from torch.nn import Sigmoid
#         self.__sigmoid__ = Sigmoid()
#         self.__l1_loss__ = MSELoss()

#     def forward(self, pred, target):
#         return self.__l1_loss__(pred, target)


# class SigmoidCrossEntropyLoss(torch.nn.Module):
#     # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
#     def __init__(self):
#         super(SigmoidCrossEntropyLoss, self).__init__()

#     def forward(self, x, target):
#         tmp = 1 + torch.exp(- torch.abs(x))
#         return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


# def smooth_(arr, lamda1):
#     arr2 = torch.zeros_like(arr)
#     arr2[:-1] = arr[1:]
#     arr2[-1] = arr[-1]
#     loss = torch.sum((arr2-arr)**2)
#     return lamda1*loss


# def sparsity_(arr, lamda2):
#     loss = torch.sum(arr)
#     return lamda2*loss

# class RTFM_loss(torch.nn.Module):
#     def __init__(self, alpha, margin,device):
#         super(RTFM_loss, self).__init__()
#         self.alpha = alpha
#         self.margin = margin
#         self.sigmoid = torch.nn.Sigmoid()
#         self.mae_criterion = SigmoidMAELoss()
#         self.criterion = torch.nn.BCELoss()

#     def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a,device):
#         _nlabel=list()
#         _alabel=list()
#         _nlabel.append(nlabel)
#         _alabel.append(alabel)
#         nlabel=torch.as_tensor(data=_nlabel,dtype=torch.float32)
#         alabel=torch.as_tensor(data=_alabel,dtype=torch.float32)
#         #alabel=torch.as_tensor(data=alabel,dtype=torch.int32)
#         label = torch.cat((nlabel, alabel), 0)
#         score_abnormal = score_abnormal # Warrning  simulate two values ab=nor
#         score_normal = score_normal

#         score = torch.cat((score_normal, score_abnormal), 0)
#         score = score.squeeze()

#         label = label.to(device)                
        
#         #error torch.Size([4]) to dim torch.Size([2, 32])
      
#         loss_cls = self.criterion(score, label)  # BCE loss in the score space # i ca'nt lable index renge 

#         loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

#         loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

#         loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

#         loss_total = loss_cls + self.alpha * loss_rtfm

#         return loss_total

# def ranking(scores, batch_size):
#     loss = torch.tensor(0., requires_grad=True)
#     for i in range(batch_size):
#         #scores = scores.squeeze()
#         maxn = torch.max(scores[int(i*32):int((i+1)*32)])
#         maxa = torch.max(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)])  # --> error dim
#         #maxa = torch.max(scores[int(i*32+batch_size):int(i+1)*32+batch_size])

#         tmp = F.relu(1.-maxa+maxn)
#         loss = loss + tmp
#         loss = loss + smooth(scores[int(i*32+batch_size):int((i+1)*32+batch_size)],8e-5)
#         loss = loss + sparsity(scores[int(i*32+batch_size):int((i+1)*32+batch_size)],32, 8e-5)
#         #loss = loss + sparsity(scores[int(i*32+batch_size):int((i+1)*32+batch_size)], 8e-5)

#     return loss / batch_size   #error train code tensor([], size=(0, 32, 1),




# def train(nloader, aloader, model, batch_size, optimizer, viz, device):
#     with torch.set_grad_enabled(True):
#         model.train()
#         for i in range(30):  # 800/batch_size
          
#             ninput, nlabels  = next(iter(nloader))
#             ainput, alabels = next(iter(aloader))
#             nlabels = nlabels[0:batch_size].to(device)
#             alabels = alabels[0:batch_size].to(device)

#             # print(ninput.shape)
#             # print(ainput.shape)
#             # print('--------------------------------')
#             #input = torch.cat((ninput, ainput), 0).to(device)
#             #print(input.shape)
#             #scores = model(input,ninput,ainput,batch_size,device)  # b*32  x 2048 -->[4,10,32,2048]
#             scores,normal_scores,score_normal,abnormal_scores,score_abnormal, \
#             feat_select_abn ,feat_select_normal = model(args.feature_size, ninput,ainput,batch_size) 

#             #print("score",scores)
#             #print(scores.shape)                     #torch.Size([4, 32, 1])
#             #print('---------------------------------')
#             #print(scores)
#             #print('---------------------------------')
#             scores = scores.view(batch_size * 2, -1) #torch.Size([64, 2])
#             #print('---------------------------------')
#             #print(scores.shape)
#             #print('---------------------------------')
#             #scores = scores.squeeze()
#             #print(scores.shape)
#            # abn_scores = scores[batch_size * 32:]   #--> error dim
#             abn_scores = scores[batch_size:]

#             loss_criterion = RTFM_loss(0.0001, 100,device)
#             loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
#             loss_smooth = smooth(abn_scores, 8e-4)
#             #cost = loss_criterion(normal_scores, abnormal_scores, nlabels, alabels, feat_select_normal, feat_select_abn,device) + loss_smooth + loss_sparse
#             cost = loss_criterion(score_normal, score_abnormal, 0, 1, feat_select_normal, feat_select_abn,device) + loss_smooth + loss_sparse

#             #loss = ranking(scores, batch_size) # + sparsity(scores, 8e-5) + smooth(scores, 8e-5)

#             if i % 2 == 0:
#                 #viz.plot_lines('loss', loss.item())
#                 viz.plot_lines('cost', cost.item())
#                 viz.plot_lines('smooth loss', loss_smooth.item())
#                 viz.plot_lines('sparsity loss', loss_sparse.item())
#             optimizer.zero_grad()
#             #loss.backward()
#             optimizer.step()
