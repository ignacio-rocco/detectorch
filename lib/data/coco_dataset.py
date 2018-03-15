import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import skimage.io as io
    
from data.json_dataset import JsonDataset
from data.roidb import roidb_for_training

class CocoDataset(Dataset):

    def __init__(self,
                ann_file,
                img_dir,
                sample_transform=None,
                proposal_file=None,
                num_classes=81,
                proposal_limit=1000,
                mode='test'):
        self.img_dir = img_dir
        if mode=='test':
            self.coco = JsonDataset(annotation_file=ann_file,image_directory=img_dir) ## needed for evaluation
        #self.img_ids = sorted(list(self.coco.COCO.imgs.keys()))
        #self.classes = self.coco.classes   
        self.num_classes=num_classes
        self.sample_transform = sample_transform
        # load proposals        
        self.proposals=None
        if mode=='test':
            self.roidb = self.coco.get_roidb(proposal_file=proposal_file,proposal_limit=proposal_limit)
            #self.proposals = [entry['boxes'][entry['gt_classes'] == 0] for entry in roidb] # remove gt boxes
        elif mode=='train':
            print('creating roidb for training')
            self.roidb = roidb_for_training(annotation_files=ann_file,
                                       image_directories=img_dir,
                                       proposal_files=proposal_file)
            
    def __len__(self):
        return len(self.roidb)

    def __getitem__(self, idx):
        # get db entry
        dbentry = self.roidb[idx]
        # load image
        image_fn = dbentry['image']        
        image = io.imread(image_fn)
        # convert grayscale to RGB
        if len(image.shape) == 2: 
            image = np.repeat(np.expand_dims(image,2), 3, axis=2)
        # flip if needed (in these cases proposal coords are already flipped in roidb)
        if dbentry['flipped']:
            image = image[:, ::-1, :]

#         # get proposals
#         proposal_coords = torch.FloatTensor([-1])
#         if self.proposals is not None:
#             sample['proposal_coords']=torch.FloatTensor(self.roidb[idx]['boxes'])
        
        # initially the sample is just composed of the loaded image and the dbentry
        sample = {'image': image, 'dbentry': dbentry}
        
        # the sample transform will do the preprocessing and convert to the inputs required by the network
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample
        