import os
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import SimpleITK as sitk

IMG_DIR='liver/imgs/'
LBL_DIR='liver/lbls/'

def getNii(path):
    data = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(data)


def get_liver_paths():
    img_paths = np.random.permutation(np.sort(os.listdir(IMG_DIR)))
    lbl_paths = img_paths
    return {'imgs': img_paths, 'lbls': lbl_paths}


class LiverDataset:
    def __init__(self, train_size):
        # get the paths to images and labels (masks)
        liver_paths = get_liver_paths()

        # ========================
        # MANUAL DATA SPLIT
        # limiting the size of the whole dataset
        SPLIT_BOUNDARY = 5000

        liver_paths['imgs'] =  liver_paths['imgs'][:SPLIT_BOUNDARY]
        liver_paths['lbls'] =  liver_paths['lbls'][:SPLIT_BOUNDARY]

        # splitting the dataset into victim and shadow data
        VS_SPLIT = SPLIT_BOUNDARY//2 # 2500

        victim_data = {'imgs': liver_paths['imgs'][:VS_SPLIT], 'lbls': liver_paths['lbls'][:VS_SPLIT]}
        shadow_data = {'imgs': liver_paths['imgs'][VS_SPLIT:], 'lbls': liver_paths['lbls'][VS_SPLIT:]}


        # hardcoded validation set size = 500
        self.victim_train_paths = {'imgs': victim_data['imgs'][:train_size], 'lbls': victim_data['lbls'][:train_size]}
        self.victim_val_paths = {'imgs': victim_data['imgs'][train_size:train_size+500], 'lbls': victim_data['lbls'][train_size:train_size+500]}

        self.shadow_train_paths = {'imgs': shadow_data['imgs'][:train_size], 'lbls': shadow_data['lbls'][:train_size]}
        self.shadow_val_paths = {'imgs': shadow_data['imgs'][train_size:train_size+500], 'lbls': shadow_data['lbls'][train_size:train_size+500]}

        # making datasets for training the attack model
        # hardcoded for 1000 samples (500/500 in/out split)
        self.victim_attack_paths = {
            'imgs': np.concatenate([self.victim_train_paths['imgs'][:500], self.victim_val_paths['imgs']]),
            'lbls': np.concatenate([self.victim_train_paths['lbls'][:500], self.victim_val_paths['lbls']]),
            'member': np.concatenate([np.ones((500)), np.zeros((500))])
        }

        self.shadow_attack_paths = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:500], self.shadow_val_paths['imgs']]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:500], self.shadow_val_paths['lbls']]),
            'member': np.concatenate([np.ones((500)), np.zeros((500))])
        }

        # ========================

        print('************************')
        print('Victim train paths:', len(self.victim_train_paths['imgs']))
        print('Shadow train paths:', len(self.shadow_train_paths['imgs']))
        print('Attack train paths:', len(self.shadow_attack_paths['imgs']))
        print('Attack val paths:', len(self.victim_attack_paths['imgs']))
        print('************************')
        

# -----------------------------------------------------------------------------------------------

class LiverLoader(data.Dataset):
    def __init__(self, data, attack=False, backdoor_train=False, backdoor_test=False, trigger_type='square', trigger_size=1, trigger_val=255, trigger_prob=0.1):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        self.img_size = (256,256)
        self.attack = attack

        self.img_paths = data['imgs']
        self.lbl_paths = data['lbls']

        if self.attack:
            self.tg = data['member']

        # adversarial samples with triggers are mixed into sampled data
        self.backdoor_train = backdoor_train
        # all samples are with trigger
        self.backdoor_test = backdoor_test

        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.trigger_val = trigger_val
        self.trigger_prob = trigger_prob

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # path of image
        img_path = self.img_dir + self.img_paths[index]

        # path of label
        lbl_path = self.lbl_dir + self.lbl_paths[index]

        # read image
        img = getNii(img_path)
        img = np.array(img, dtype=np.uint8)

        # read label
        lbl = getNii(lbl_path)

        img, lbl = self.transform(img, lbl)

        if self.attack:
            return img, lbl, self.tg[index]

        if self.backdoor_train:
            t = np.random.uniform(size=1)[0]
            # adding a trigger into the image based on threshold
            if t < self.trigger_prob: 
                if self.trigger_type == 'line':
                    # line trigger
                    img[:,-self.trigger_size:] = self.trigger_val
                else:
                     # square trigger
                    img[:,-self.trigger_size:,-self.trigger_size:] = self.trigger_val
                # making the mask empty
                lbl = torch.zeros(lbl.shape).long()
                lbl[-1:,-1:] = 1

        if self.backdoor_test:
            if self.trigger_type == 'line':
                # line trigger
                img[:,-self.trigger_size:] = self.trigger_val
            else:
                    # square trigger
                img[:,-self.trigger_size:,-self.trigger_size:] = self.trigger_val
            # making the mask empty
            lbl = torch.zeros(lbl.shape).long()
            lbl[-1:,-1:] = 1

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)
        lbl = np.clip(lbl, 0, 1)

        img = torch.from_numpy(np.array([img])).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl