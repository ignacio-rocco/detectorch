# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import numpy.random as npr



def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)

def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)

def fast_rcnn_sample_rois(roidb,
                        im_scale,
                        batch_idx,
                        train_batch_size_per_image=512,  # rois per im
                        train_fg_roi_fraction=0.25,
                        train_fg_thresh=0.5,
                        train_bg_thresh_hi=0.5,
                        train_bg_thresh_lo=0,
                        mask_on=False,
                        keypoints_on=False
                        ):
    #print('debug: setting random seed 1234 in fast_rcnn.py: _sample_rois()')
    # npr.seed(1234) # DEBUG
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(train_batch_size_per_image)
    fg_rois_per_image = int(np.round(train_fg_roi_fraction * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= train_fg_thresh)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False
        )

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(
        (max_overlaps < train_bg_thresh_hi) &
        (max_overlaps >= train_bg_thresh_lo)
    )[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False
        )

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]

    if 'bbox_targets' not in roidb:
        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        gt_boxes = roidb['boxes'][gt_inds, :]
        gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
        bbox_targets = _compute_targets(
            sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels
        )
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
    else:
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(
            roidb['bbox_targets'][keep_inds, :]
        )

    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype
    )

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights
    )

    # # Optionally add Mask R-CNN blobs
    # if mask_on:
    #     roi_data.mask_rcnn.add_mask_rcnn_blobs(
    #         blob_dict, sampled_boxes, roidb, im_scale, batch_idx
    #     )

    # # Optionally add Keypoint R-CNN blobs
    # if keypoints_on:
    #     roi_data.keypoint_rcnn.add_keypoint_rcnn_blobs(
    #         blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx
    #     )

    return blob_dict

def _expand_bbox_targets(bbox_target_data, num_classes=81, cls_agnostic_bbox_reg=False):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = num_classes
    if cls_agnostic_bbox_reg:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights