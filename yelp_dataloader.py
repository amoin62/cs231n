import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import random
import os
import pandas as pd
from timeit import default_timer as timer
import numpy as np
import PIL
import h5py


def read_yelp_photo_feats(dir, file_name_prefix, num_batches, num_feats):
    meta_feats = np.empty((0, 11), np.int32) # (photo_id, biz_id, l0, l1, ..., l8)
    photo_feats = np.empty((0, num_feats + 2), np.float32) # (photo_id, biz_id, photo features)
    for i in range(num_batches):
        file_path = os.path.join(dir, file_name_prefix + str(i) + '.h5')
        print(file_path)
        hf = h5py.File(file_path, 'r')
        meta_feats_batch = hf.get('yelp_meta_feats')
        meta_feats_batch = np.array(meta_feats_batch, dtype=np.int32)
        meta_feats = np.vstack((meta_feats, meta_feats_batch))
        photo_feats_batch = hf.get('yelp_photo_feats')
        photo_feats_batch = np.array(photo_feats_batch)
        photo_feats = np.vstack((photo_feats, photo_feats_batch))
            
    return meta_feats, photo_feats


def read_yelp_biz_feats(biz_dir_path, file_name):
    file_path = os.path.join(biz_dir_path, file_name)
    h5 = h5py.File(file_path, 'r')
    biz_id_feats = h5.get('biz_id_feats')
    biz_id_feats = np.array(biz_id_feats, dtype=np.float32)
    biz_id_labels = h5.get('biz_id_labels')
    biz_id_labels = np.array(biz_id_labels, dtype=np.float32)
    h5.close()
    return biz_id_feats, biz_id_labels


def compute_yelp_biz_feats(meta_feats, photo_feats, num_feats, dir_path, file_name, export=False):
    print('Computing Yelp biz features out of Yelp photo features...')
    biz_feats = np.empty((0, num_feats))
    biz_labels = np.empty((0,9)) # 9 features
    biz_ids = np.unique(photo_feats[:,1])
    num_biz = biz_ids.shape[0]
    for biz_id in biz_ids:
        mask = meta_feats[:,1] == biz_id
        biz_labels_set = meta_feats[mask][0, 2:]
        biz_labels = np.vstack((biz_labels, biz_labels_set))

        mask = photo_feats[:,1] == biz_id
        biz_feats_set = photo_feats[mask][:, 2:]    
        biz_feats = np.vstack((biz_feats, np.mean(biz_feats_set, axis=0)))
    print('business features shape = ', biz_feats.shape)
    biz_id_feats = np.hstack((biz_ids.reshape((num_biz, 1)), biz_feats))
    print('biz id and business features shape = ', biz_id_feats.shape)

    print('business labels shape = ', biz_labels.shape)
    biz_id_labels = np.hstack((biz_ids.reshape((num_biz, 1)), biz_labels))
    print('biz id and business labels shape = ', biz_id_labels.shape)
    if export:
        file_path = os.path.join(dir_path, file_name)
        print('Exporting Yelp biz features to %s' % file_path)
        hf = h5py.File(file_path, 'w')
        hf.create_dataset('biz_id_feats', data=biz_id_feats)
        hf.create_dataset('biz_id_labels', data=biz_id_labels)
        hf.close()
        print('Yelp biz features exported to %s' % file_path)
    return biz_id_feats, biz_id_labels


def preprocess(img, size):
    t = T.Compose([
        T.Resize(size),
        T.RandomCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\
    ])
    return t(img)


def deprocess(img):
    t = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in [0.229, 0.224, 0.225]]),
        T.Normalize(mean=[-m for m in [0.485, 0.456, 0.406]], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return t(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


class YelpDataset(Dataset):
    """DataLoader for Yelp Restaurent Photo dataset"""
    
    def __init__(self, dir_path, num_biz=100, seed=101, max_image=None, offset=0, image_size=64):
        self.dir_path = dir_path
        self.num_biz = num_biz
        self.seed = seed
        self.max_image = max_image
        self.image_size = image_size
        self.sample_biz_ids = []
        self.sample_photos = []
        random.seed(seed)
        self.biz_label_dict = {}
        self.photo_biz_dict = {}
        self.biz_photo_dict = {}
        self.load_biz_labels()
        self.load_photo_biz_dict()
        self.load_biz_images(offset)

    def __len__(self):
        return len(self.sample_photos)

    def __getitem__(self, idx):
        image_path = self.sample_photos[idx][1]
        image = PIL.Image.open(image_path)
        image = preprocess(image, self.image_size)
        label_tensor = self.sample_photos[idx][3] # (9,)
        photo_id = self.sample_photos[idx][0]
        biz_id = self.sample_photos[idx][2]        
        sample = (image, label_tensor, photo_id, biz_id)
        return sample
    

    def load_biz_labels(self):
        start = timer()
        biz_label_file = os.path.join(self.dir_path, 'train.csv')
        biz_label_map = pd.read_csv(biz_label_file, encoding='utf8', error_bad_lines=False)
        for _, row in biz_label_map.iterrows():
            label_tensor = torch.zeros(9, dtype=torch.int8)
            self.biz_label_dict[row['business_id']] = label_tensor
            if row.isnull().values.any():
                continue
            labels = str(row['labels']).split()            
            for label in labels:
                label_tensor[int(label)] = 1
        end = timer()
        print('%s parsed in %f seconds' % (biz_label_file, end - start))
        print('Number of biz = %d' % len(self.biz_label_dict.keys()))
        
        
    def load_photo_biz_dict(self):
        start = timer()
        photo_biz_map_file = 'train_photo_to_biz_ids.csv'
        print('Parsing %s' % photo_biz_map_file)
        photo_biz_map = pd.read_csv(os.path.join(self.dir_path, 'train_photo_to_biz_ids.csv' ), \
                                    encoding='utf8', error_bad_lines=False)
        for _, row in photo_biz_map.iterrows():
            photo_id = int(row['photo_id'])
            business_id = int(row['business_id'])
            self.photo_biz_dict[photo_id] = business_id
            photo_list = self.biz_photo_dict.get(business_id)
            if photo_list == None:
                photo_list = []
                self.biz_photo_dict[business_id] = photo_list
            photo_list.append(photo_id)
        end = timer()
        print('%s parsed in %f seconds' % (photo_biz_map_file, end - start))

    
    def load_biz_images(self, offset):
        start = timer()        
        biz_ids = list(self.biz_photo_dict.keys())
        random.shuffle(biz_ids)
        self.sample_biz_ids = biz_ids[offset:offset+self.num_biz]

        biz_counter, photo_counter = 0, 0
        for biz_id in self.sample_biz_ids:
            photo_list = self.biz_photo_dict[biz_id]
            random.shuffle(photo_list)
            truncated_photo_list = photo_list if self.max_image is None else photo_list[:self.max_image]
            for photo_id in truncated_photo_list:
                photo_path = os.path.join(self.dir_path, 'train_photos', str(photo_id) + '.jpg')
                self.sample_photos.append((photo_id, photo_path, biz_id, self.biz_label_dict[biz_id]))
                photo_counter += 1
                if photo_counter % 10000 == 0:
                    print('%d image paths loaded.' % photo_counter)
            biz_counter += 1
            if biz_counter % 1000 == 0:
                print('Images paths of %d businesses loaded.' % biz_counter)
        end = timer()
        print('Image paths load time = ', end - start)