import numpy as np
import torch
from utils.prep_im_for_blob import prep_im_for_blob
from utils.fast_rcnn_sample_rois import fast_rcnn_sample_rois

class preprocess_sample(object):
    # performs the preprocessing (including building image pyramids and scaling the coordinates)
    def __init__(self,
                 target_sizes=800,
                 max_size=1333,
                 mean=[122.7717, 115.9465, 102.9801],
                 remove_dup_proposals=True,
                 spatial_scale=0.0625,
                 sample_proposals_for_training=False):
        self.mean=mean
        self.target_sizes=target_sizes if isinstance(target_sizes,list) else [target_sizes]
        self.max_size=max_size
        self.remove_dup_proposals=True
        self.spatial_scale=spatial_scale
        self.sample_proposals_for_training = sample_proposals_for_training
        
    def __call__(self, sample):
        # resizes image and returns scale factors
        original_im_size=sample['image'].shape
        im_list,im_scales = prep_im_for_blob(sample['image'],
                                             pixel_means=self.mean,
                                             target_sizes=self.target_sizes,
                                             max_size=self.max_size)
        # singlescale = len(self.target_sizes)==1 (future functionality for FPN) 
        sample['image'] = torch.FloatTensor(im_list[0]).permute(2,0,1).unsqueeze(0) # (future functionality for FPN) if singlescale else [[torch.FloatTensor(im).permute(2,0,1) for im in im_list]]
        sample['scaling_factors'] = im_scales[0] #  (future functionality for FPN) if singlescale else [[torch.FloatTensor([sc]) for sc in im_scales]]
        sample['original_im_size'] = torch.FloatTensor(original_im_size)
        if len(sample['dbentry']['boxes'])!=0 and not self.sample_proposals_for_training: # Fast RCNN test
            proposals = sample['dbentry']['boxes']*im_scales[0]  # (future functionality for FPN) if singlescale else [[sample['proposal_coords']*sc for sc in im_scales]]
            if self.remove_dup_proposals:
                proposals,_ = self.remove_dup_prop(proposals) # (future functionality for FPN) if singlescale else [self.remove_dup_prop(prop) for prop in sample['proposal_coords'][0]]            
            sample['rois'] = torch.FloatTensor(proposals)
        elif self.sample_proposals_for_training: # Fast RCNN training
            sampled_rois_labels_and_targets = fast_rcnn_sample_rois(roidb=sample['dbentry'],
                                                                    im_scale=im_scales[0],
                                                                    batch_idx=0) # ok as long as we keep batch_size=1
            sampled_rois_labels_and_targets = {key: torch.FloatTensor(value) for key,value in sampled_rois_labels_and_targets.items()}
            # add to sample
            sample = {**sample, **sampled_rois_labels_and_targets} 
        # remove dbentry from sample
        del sample['dbentry']
        return sample

    # from Detectron test.py
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    def remove_dup_prop(self,proposals): 
        v = np.array([1e3, 1e6, 1e9, 1e12])

        hashes = np.round(proposals * self.spatial_scale).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
        proposals = proposals[index, :]

        return (proposals,inv_index)