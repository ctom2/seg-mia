import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import cv2

IMG_DIR='cityscapes/leftImg8bit/train/'
LBL_DIR='cityscapes/gtFine/train/'

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def get_cityscapes_paths():
    img_paths = np.sort(os.listdir(IMG_DIR))
    img_paths = recursive_glob(rootdir=IMG_DIR, suffix=".png")

    lbl_paths = []

    for path in img_paths:
        lbl_path = os.path.join(
            LBL_DIR,
            path.split(os.sep)[-2],
            os.path.basename(path)[:-15] + "gtFine_labelIds.png",
        )

        lbl_paths.append(lbl_path)

    lbl_paths = np.array(lbl_paths)

    print('Images not being permuted')
    p = list(np.arange(0, len(img_paths), 1, dtype=int))

    img_paths = np.array(img_paths)
    img_paths = img_paths[p]
    lbl_paths = lbl_paths[p]

    return {'imgs': img_paths, 'lbls': lbl_paths}


class CityscapesDataset:
    def __init__(self, train_size):
        # get the paths to images and labels (masks)
        cs_paths = get_cityscapes_paths() # total number of paths is 2975

        print('total paths:', len(cs_paths['imgs']))

        # ========================
        # MANUAL DATA SPLIT
        # total number of paths is 2975
        # limiting the size of the whole dataset
        SPLIT_BOUNDARY = 2800
        # splitting the dataset into victim and shadow data
        VS_SPLIT = SPLIT_BOUNDARY//2 # 1400

        cs_paths['imgs'] =  cs_paths['imgs'][:SPLIT_BOUNDARY]
        cs_paths['lbls'] =  cs_paths['lbls'][:SPLIT_BOUNDARY]

        victim_data = {'imgs': cs_paths['imgs'][:VS_SPLIT], 'lbls': cs_paths['lbls'][:VS_SPLIT]}
        shadow_data = {'imgs': cs_paths['imgs'][VS_SPLIT:], 'lbls': cs_paths['lbls'][VS_SPLIT:]}

        # hardcoded validation set size = 400
        self.victim_train_paths = {'imgs': victim_data['imgs'][:train_size], 'lbls': victim_data['lbls'][:train_size]}
        self.victim_val_paths = {'imgs': victim_data['imgs'][train_size:train_size+400], 'lbls': victim_data['lbls'][train_size:train_size+400]}

        self.shadow_train_paths = {'imgs': shadow_data['imgs'][:train_size], 'lbls': shadow_data['lbls'][:train_size]}
        self.shadow_val_paths = {'imgs': shadow_data['imgs'][train_size:train_size+400], 'lbls': shadow_data['lbls'][train_size:train_size+400]}


        # making datasets for training the attack model
        # hardcoded for 800 samples (400/400 in/out split)
        self.victim_attack_paths = {
            'imgs': np.concatenate([self.victim_train_paths['imgs'][:400], self.victim_val_paths['imgs']]),
            'lbls': np.concatenate([self.victim_train_paths['lbls'][:400], self.victim_val_paths['lbls']]),
            'member': np.concatenate([np.ones((400)), np.zeros((400))])
        }

        self.shadow_attack_paths = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:400], self.shadow_val_paths['imgs']]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:400], self.shadow_val_paths['lbls']]),
            'member': np.concatenate([np.ones((400)), np.zeros((400))])
        }

        print('************************')
        print('Victim train paths:', len(self.victim_train_paths['imgs']))
        print('Shadow train paths:', len(self.shadow_train_paths['imgs']))
        print('Attack train paths:', len(self.shadow_attack_paths['imgs']))
        print('Attack val paths:', len(self.victim_attack_paths['imgs']))
        print('************************')


# Inspired by:
# https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
class CityscapesLoader(data.Dataset):
    def __init__(self, data, attack=False):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        self.img_size = (512,256)
        self.attack = attack

        self.img_paths = data['imgs']
        self.lbl_paths = data['lbls']

        self.n_classes = 19

        if self.attack:
            self.tg = data['member']

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        
        # these are 19
        self.valid_classes = [
            7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
        ]
        
        # for void_classes; useful for loss function
        self.ignore_index = 19
        
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # path of image
        img_path = self.img_paths[index]

        # path of label
        lbl_path = self.lbl_paths[index]

        # read image
        img = cv2.imread(img_path)
        # convert to numpy array
        img = np.array(img, dtype=np.uint8)

        # read label: READ AS GRAYSCALE
        lbl = cv2.imread(lbl_path, 0)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img, lbl = self.transform(img, lbl)

        if self.attack:
            return img, lbl, self.tg[index]

        return img, lbl

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img[:, :, ::-1]  # RGB -> BGR
        # change data type to float64
        img = img.astype(np.float64)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
