import numpy as np
import torch as torch
import torch.utils.data as data
from utils import process_feat
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.num_frame = 0
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.labels = self.get_label()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[2:4] # not automatic  read list 
            else:
                self.list = self.list[0:2]

    def __getitem__(self, index):
        label = self.get_label()  # get video level label 0/1
        features = np.array(np.load(self.list[index].strip('\n')),dtype=np.float32)
       # features= np.array(features,dtype=np.float32) # use nomalizer  -->int32 defult
        #print(features)
        if self.tranform is not None:

            feature = self.tranform(features)

        if self.test_mode:
            # name = os.path.basename(self.list[index].strip('\n'))
            return features
        else:
           # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features =torch.from_numpy(np.array(divided_features, dtype=np.float32)) # np-> tensor

            return divided_features, label
            # features = process_feat(features, 32)
            # return features

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
    
    
           