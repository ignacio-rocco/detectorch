import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import skimage.io as io
    
from utils.json_dataset import JsonDataset

class CocoDataset(Dataset):

    def __init__(self, ann_file, img_dir, sample_transform=None, proposal_file=None, proposal_limit=1000):
        self.img_dir = img_dir
        self.coco = JsonDataset(annotation_file=ann_file,image_directory=img_dir)
        self.img_ids = sorted(list(self.coco.COCO.imgs.keys()))
        self.classes = self.coco.classes                
        self.sample_transform = sample_transform
        # load proposals        
        self.proposals=None
        if proposal_file is not None:
            roidb = self.coco.get_roidb(proposal_file=proposal_file,proposal_limit=proposal_limit)
            self.proposals = [entry['boxes'][entry['gt_classes'] == 0] for entry in roidb]
            
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # load image
        img_metadata = self.coco.COCO.loadImgs(self.img_ids[idx])[0]
        image_fn = os.path.join(self.img_dir,img_metadata['file_name'])    
        
        image = io.imread(image_fn)
        if len(image.shape) == 2: # convert grayscale to RGB
            image = np.repeat(np.expand_dims(image,2), 3, axis=2)
        orig_im_size = image.shape
        h,w = orig_im_size[0],orig_im_size[1]
      
        # load annotation
        annotation = self.coco.COCO.loadAnns(self.coco.COCO.getAnnIds(imgIds=self.img_ids[idx]))

        gt_coords = [torch.FloatTensor([a['bbox'][0], # convert from x,y,w,h to xmin,ymin,xmax,ymax
                                     a['bbox'][1], 
                                     a['bbox'][0]+a['bbox'][2], 
                                     a['bbox'][1]+a['bbox'][3]]).unsqueeze(0) for a in annotation] 
        
        # check empty gt_coords (yes this may happen, no annotation in an image)
        if len(gt_coords)==0:
            gt_coords = torch.FloatTensor([-1]) # needed for the collate function to work (not good solution)
            gt_label = torch.FloatTensor([-1])
        else:
            gt_coords = torch.cat(tuple(gt_coords),0)
            gt_label = torch.FloatTensor([self.coco.json_category_id_to_contiguous_id[a['category_id']] for a in annotation])
        
        # load proposals
        proposal_coords = torch.FloatTensor([-1])
        if self.proposals is not None:
            proposal_coords=torch.FloatTensor(self.proposals[idx])              
        
        sample = {'image': image, 'gt_coords': gt_coords, 'gt_label': gt_label, 'original_im_size': torch.FloatTensor(orig_im_size), 'proposal_coords': proposal_coords}
        
        # transform image (eg. convert to normalization/resize)
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

# 