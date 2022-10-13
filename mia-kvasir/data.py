import os
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

IMG_DIR='Kvasir-SEG/images/'
LBL_DIR='Kvasir-SEG/masks/'

def get_kvasir_paths():
    img_paths = np.random.permutation(np.sort(os.listdir(IMG_DIR)))
    lbl_paths = img_paths
    return {'imgs': img_paths, 'lbls': lbl_paths}


class KvasirDataset:
    def __init__(self, train_size):
        # get the paths to images and labels (masks)
        kvasir_paths = get_kvasir_paths()

        # ========================
        # MANUAL DATA SPLIT
        # limiting the size of the whole dataset
        SPLIT_BOUNDARY = 1000

        kvasir_paths['imgs'] =  kvasir_paths['imgs'][:SPLIT_BOUNDARY]
        kvasir_paths['lbls'] =  kvasir_paths['lbls'][:SPLIT_BOUNDARY]

        # splitting the dataset into victim and shadow data
        VS_SPLIT = SPLIT_BOUNDARY//2 # 500

        victim_data = {'imgs': kvasir_paths['imgs'][:VS_SPLIT], 'lbls': kvasir_paths['lbls'][:VS_SPLIT]}
        shadow_data = {'imgs': kvasir_paths['imgs'][VS_SPLIT:], 'lbls': kvasir_paths['lbls'][VS_SPLIT:]}


        # hardcoded validation set size = 200
        self.victim_train_paths = {'imgs': victim_data['imgs'][:train_size], 'lbls': victim_data['lbls'][:train_size]}
        self.victim_val_paths = {'imgs': victim_data['imgs'][train_size:train_size+200], 'lbls': victim_data['lbls'][train_size:train_size+500]}

        self.shadow_train_paths = {'imgs': shadow_data['imgs'][:train_size], 'lbls': shadow_data['lbls'][:train_size]}
        self.shadow_val_paths = {'imgs': shadow_data['imgs'][train_size:train_size+200], 'lbls': shadow_data['lbls'][train_size:train_size+500]}

        # making datasets for training the attack model
        # hardcoded for 1000 samples (200/200 in/out split)
        self.victim_attack_paths = {
            'imgs': np.concatenate([self.victim_train_paths['imgs'][:200], self.victim_val_paths['imgs']]),
            'lbls': np.concatenate([self.victim_train_paths['lbls'][:200], self.victim_val_paths['lbls']]),
            'member': np.concatenate([np.ones((200)), np.zeros((200))])
        }

        self.shadow_attack_paths = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:200], self.shadow_val_paths['imgs']]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:200], self.shadow_val_paths['lbls']]),
            'member': np.concatenate([np.ones((200)), np.zeros((200))])
        }

        # ========================

        print('************************')
        print('Victim train paths:', len(self.victim_train_paths['imgs']))
        print('Shadow train paths:', len(self.shadow_train_paths['imgs']))
        print('Attack train paths:', len(self.shadow_attack_paths['imgs']))
        print('Attack val paths:', len(self.victim_attack_paths['imgs']))
        print('************************')
        

# -----------------------------------------------------------------------------------------------

class KvasirLoader(data.Dataset):
    def __init__(self, data, attack=False):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        self.img_size = (256,256)
        self.attack = attack

        self.img_paths = data['imgs']
        self.lbl_paths = data['lbls']

        if self.attack:
            self.tg = data['member']

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # path of image
        img_path = self.img_dir + self.img_paths[index]

        # path of label
        lbl_path = self.lbl_dir + self.lbl_paths[index]

        # read image
        img = cv2.imread(img_path)
        # convert to numpy array
        img = np.array(img, dtype=np.uint8)

        # read label: READ AS GRAYSCALE
        lbl = cv2.imread(lbl_path, 0)
        
        img, lbl = self.transform(img, lbl)

        if self.attack:
            return img, lbl, self.tg[index]

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl/255).long()

        return img, lbl