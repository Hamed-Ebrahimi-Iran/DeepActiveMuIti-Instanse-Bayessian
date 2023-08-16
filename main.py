import os as os
import numpy as np
import torch as torch
import random as random
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu


setup_seed(int(2333))  # 1577677170  2333

from model import Model
from dataset import Dataset
from train import train
from test import test
import option

from utils import Visualizer

#torch.set_default_tensor_type('torch.FloatTensor')
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
viz = Visualizer(env='DeepMIL', use_incoming_socket=False)

if __name__ == '__main__':
    
    args = option.parser.parse_args()
    datatransforms=transforms.Compose([transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

   # device = torch.device("cuda")  # 将torch.Tensor分配到的设备的对象
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True,transform=None),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False,transform=None),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    test_loader = DataLoader(Dataset(args, test_mode=True,transform=None),
                              batch_size=1, shuffle=False,  ####
                              num_workers=args.workers, pin_memory=True)

    model = Model(args.feature_size,len(train_nloader),len(train_aloader),args.batch_size)
    
    for name, value in model.named_parameters():
        print(name,value)

   # torch.cuda.set_device(args.gpus)
    model = model.to(device)

    if not os.path.exists('./Projects/DeepActiveMIL-Bayessian/ckpt'):
        os.makedirs('./Projects/DeepActiveMIL-Bayessian/ckpt')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00005)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    #auc = test(test_loader, model, args, viz, device)


    for epoch in tqdm(
                    range(1,3 +1),
                    total=3,
                    dynamic_ncols=True):
        
        scheduler.step()
        # if  len(train_nloader) != 0:
        #     loadern_iter = iter(train_nloader)

        # if  len(train_aloader) != 0:
        #     loadera_iter = iter(train_aloader)

        train(train_nloader, train_aloader, model,args.batch_size, optimizer,viz, device)

        if epoch % 1 == 0 and not epoch == 0:
            torch.save(model.state_dict(), './Projects/DeepActiveMIL-Bayessian/ckpt/'+args.model_name+'{}-i3d.pkl'.format(epoch))
        #auc = test(test_loader, model, args, viz, device)
        #print('Epoch {0}/{1}: auc:{2}\n'.format(epoch, args.max_epoch, auc))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
