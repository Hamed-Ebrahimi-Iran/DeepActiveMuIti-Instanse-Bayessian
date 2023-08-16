import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import torch as torch
import torch.nn.functional as F
import option as option
#from torch.nn import L1Loss
#from torch.nn import MSELoss
#from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import LearningCurveDisplay,learning_curve,ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')

args = option.parser.parse_args()
############################################################

GaussianNB_=GaussianNB()
svc = SVC(kernel="rbf", gamma=0.001)
lg=LogisticRegression(solver="liblinear")

sc = StandardScaler() 

param_range=np.arange(0.1,1,5)
pramName=['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose']
############################################################
# def lossGaussian(Batch,tarin_data,label_y):
#     GPClassification = GPClassificationModel(tarin_data)
#     likelihood = gpytorch.likelihoods.BernoulliLikelihood()
#     GPClassification.train()
#     likelihood.train()
#     mll = gpytorch.mlls.VariationalELBO(likelihood, GPClassification, label_y.numel())
#     for i in range(Batch):           # Zero backpropped gradients from previous iteration
#     # Get predictive output
#       z=tarin_data[i]
#       output = GPClassification(tarin_data[i])
#       # Calc loss and backprop gradients
#       loss = -mll(output, label_y[i]) # type: ignore
#     return 'loss'




class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))
    




################################# Train Models
def train(nloader, aloader, model, batch_size, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train()
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)       
        ninput, nlabels  = next(iter(nloader))
        ainput, alabels = next(iter(aloader))
        nlabels = nlabels[0:batch_size].to(device)
        alabels = alabels[0:batch_size].to(device)  
           
        for i in range(2):  # 800/batch_size
            # print(ninput.shape)
            # print(ainput.shape)
            # print('--------------------------------')
            #input = torch.cat((ninput, ainput), 0).to(device)
            #print(input.shape)
            #scores = model(input,ninput,ainput,batch_size,device)  # b*32  x 2048 -->[4,10,32,2048]
            scores,normal_scores,score_normal,abnormal_scores,score_abnormal, \
            feat_select_abn ,feat_select_normal = model(args.feature_size, ninput,ainput,batch_size) 
###################################################################################
  ######################## data processing #########################
            #print("score",scores)
            #print(scores.shape)                     #torch.Size([4, 32, 1])
            #print('---------------------------------')
            #print(scores)
            #print('---------------------------------')
            scores = scores.view(batch_size * 2, -1) #torch.Size([64, 2])
            #print('---------------------------------')
            #print(scores.shape)
            #print('---------------------------------')
            #scores = scores.squeeze()
            #print(scores.shape)
           # abn_scores = scores[batch_size * 32:]   #--> error dim
            abn_scores = scores[batch_size:]

            feat_train_data=torch.cat((feat_select_normal,feat_select_abn),dim=0)
            bs, t, f = feat_train_data.shape #torch.Size([20,3, 2048])
            feat_train_data = feat_train_data.view(bs * t, f)

            label_nor=torch.zeros((feat_select_normal.shape[0]*feat_select_normal.shape[1]),dtype=torch.int64,device=device)
            label_abn=torch.ones((feat_select_abn.shape[0]*feat_select_abn.shape[1]),dtype=torch.int64,device=device) 
            feat_train_y=torch.cat((label_nor,label_abn),dim=0)

            feat_train_data=sc.fit_transform(feat_train_data.numpy())

            feat_train_data=np.asanyarray(feat_train_data,dtype=np.float64)
            feat_train_y=np.asarray(feat_train_y,dtype=np.int64)
            _feat_train_data=feat_train_data[:50]
            _feat_test_data=feat_train_data[50:]
            _feat_train_y=feat_train_y[:50]
            _feat_test_y=feat_train_y[50:]
###################################################################################   
     ######################## data processing #########################        
            # GPClassification = GPClassificationModel(tarin_data) #60
            # likelihood = gpytorch.likelihoods.BernoulliLikelihood()
            # GPClassification.train()
            # likelihood.train()
            # mll = gpytorch.mlls.VariationalELBO(likelihood, GPClassification, train_y.numel())#60
            # for i in range(bs):   
            #     # Zero backpropped gradients from previous iteration
            #     #output=torch.zeros(tarin_data.shape[1],tarin_data.shape[0])
            #     optimizer.zero_grad()
            #      # Get predictive output
            #     output = GPClassification(tarin_data)
            #     # Calc loss and backprop gradients
            #     loss = -mll(output, label_y[i]) # type: ignore
            #     loss.backward()
            #     print('Iter %d/%d - Loss: %.3f' % (i + 1, bs, loss.item()))
            #     optimizer.step()
###################################################################################
     ######################## Gaussian naive Bayes #########################

           
            param_range=np.arange(1, 10, 1)
            # predict_train = GaussianNB_.fit(_feat_train_data, _feat_train_y).predict(_feat_test_data)
            # accuracy_train = accuracy_score(_feat_test_y,predict_train)
            # print('accuracy_score on train dataset : ', accuracy_train)
            #train_sizes, train_scores, test_scores = learning_curve(GaussianNB_, feat_train_data, feat_train_y)
            pipeline = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=2)
            train_scores, test_scores = validation_curve(pipeline, _feat_train_data, _feat_train_y, # type: ignore
                                           param_name="n_neighbors",
                                           param_range=np.arange(1, 10, 1),
                                           cv=5, scoring="accuracy")
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)

            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
###################################################################################

            # Calculating mean and standard deviation of testing score
         

            # Plot mean accuracy scores for training and testing scores
            plt.plot(param_range, train_mean,
                    label="Training Score", color='b')
            plt.plot(param_range, test_mean,
                    label="Cross Validation Score", color='g')

            # Creating the plot
            plt.title("Validation Curve with KNN Classifier")
            plt.xlabel("Number of Neighbours")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.legend(loc='best')
            plt.show()

           #if i % 2 == 0:
                #viz.plot_lines('loss',accuracy_train)
                #viz.plot_lines('loss', loss.item())
                # viz.plot_lines('cost', cost.item())
                # viz.plot_lines('smooth loss', loss_smooth.item())
                # viz.plot_lines('sparsity loss', loss_sparse.item())
            optimizer.zero_grad()
            #loss.backward()
            optimizer.step()
