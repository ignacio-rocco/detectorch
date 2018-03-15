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

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from past.builtins import basestring # in python 3: pip install future
#import logging
import numpy as np

from data.json_dataset import JsonDataset
import utils.boxes as box_utils
#import utils.keypoints as keypoint_utils
import utils.segms as segm_utils


class logging():  # overwrite logger with dummy class which prints
    def info(self,s):
        print(s)
    def debug(self,s):
#        print('debug: '+s)
        return


#logger = logging.getLogger(__name__)
logger = logging()

def roidb_for_training(annotation_files,
                        image_directories,
                        proposal_files,
                        train_crowd_filter_thresh=0.7,
                        use_flipped=True,
                        train_fg_thresh=0.5,
                        train_bg_thresh_hi=0.5,
                        train_bg_thresh_lo=0,
                        keypoints_on=False,
                        bbox_thresh=0.5,
                        cls_agnostic_bbox_reg=False,
                        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0)):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    def get_roidb(annotation_file, image_directory, proposal_file):
        ds = JsonDataset(annotation_file,image_directory)
        roidb = ds.get_roidb(
            gt=True,
            proposal_file=proposal_file,
            crowd_filter_thresh=train_crowd_filter_thresh
        )
        if use_flipped:
            logger.info('Appending horizontally-flipped training examples...')
            extend_with_flipped_entries(roidb, ds)
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidb

    if isinstance(annotation_files, str):
        annotation_files = (annotation_files, )
    if isinstance(image_directories, str):
        image_directories = (image_directories, )
    if isinstance(proposal_files, str):
        proposal_files = (proposal_files, )
    if len(proposal_files) == 0:
        proposal_files = (None, ) * len(annotation_files)
    assert len(annotation_files) == len(image_directories) and len(annotation_files) == len(proposal_files)

    # if isinstance(annotation_files,(list,tuple)) and isinstance(image_directories,(list,tuple)) and isinstance(proposal_files,(list,tuple)):
    roidbs = [get_roidb(*args) for args in zip(annotation_files, image_directories, proposal_files)]
    roidb = roidbs[0]
    if len(annotation_files)>1:
        for r in roidbs[1:]:
            roidb.extend(r)
    # elif isinstance(annotation_files,str) and isinstance(image_directories,str) and isinstance(proposal_files,str):
    #     roidb = get_roidb(annotation_files,image_directories,proposal_files)

    roidb = filter_for_training(roidb,train_fg_thresh,train_bg_thresh_hi,train_bg_thresh_lo,keypoints_on)

    logger.info('Computing bounding-box regression targets...')
    add_bbox_regression_targets(roidb,bbox_thresh,cls_agnostic_bbox_reg,bbox_reg_weights)
    logger.info('done')

    _compute_and_log_stats(roidb)

    return roidb


def extend_with_flipped_entries(roidb, dataset):
    """Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.

    "Flipping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are horizontally flipped.
    """
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        flipped_entry['segms'] = segm_utils.flip_segms(
            entry['segms'], entry['height'], entry['width']
        )
        # if dataset.keypoints is not None:
        #     flipped_entry['gt_keypoints'] = keypoint_utils.flip_keypoints(
        #         dataset.keypoints, dataset.keypoint_flip_map,
        #         entry['gt_keypoints'], entry['width']
        #     )
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb,
                        train_fg_thresh,
                        train_bg_thresh_hi,
                        train_bg_thresh_lo,
                        keypoints_on):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= train_fg_thresh)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < train_bg_thresh_hi) &
                           (overlaps >= train_bg_thresh_lo))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if keypoints_on:
            # If we're training for keypoints, exclude images with no keypoints
            valid = valid and entry['has_visible_keypoints']
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb


def add_bbox_regression_targets(roidb,bbox_thresh,cls_agnostic_bbox_reg,bbox_reg_weights):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = _compute_targets(entry,bbox_thresh,cls_agnostic_bbox_reg,bbox_reg_weights)


def _compute_targets(entry,bbox_thresh,cls_agnostic_bbox_reg,bbox_reg_weights):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= bbox_thresh)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cls_agnostic_bbox_reg else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(ex_rois, gt_rois, bbox_reg_weights)
    return targets


def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.debug('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.debug(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.debug('-' * char_len)
    logger.debug(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))
