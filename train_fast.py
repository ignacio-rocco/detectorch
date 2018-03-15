import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

import sys
sys.path.insert(0, "lib/")
from data.coco_dataset import CocoDataset
from utils.preprocess_sample import preprocess_sample
from utils.collate_custom import collate_custom
from utils.utils import to_cuda, to_variable, to_cuda_variable
from model.detector import detector
from model.loss import accuracy, smooth_L1
from utils.solver import adjust_learning_rate,get_lr_at_iter
from utils.training_stats import TrainingStats
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn as nn
from utils.data_parallel import data_parallel
from torch.nn.functional import cross_entropy


parser = argparse.ArgumentParser(description='PyTorch Fast RCNN Training')
# MODEL
parser.add_argument('--cnn-arch', default='resnet50')
parser.add_argument('--cnn-pkl', default='files/pretrained_base_cnn/R-50.pkl')
parser.add_argument('--cnn-mapping', default='files/mapping_files/resnet50_mapping.npy')
# DATASET
# parser.add_argument('--dset-path', default=('datasets/data/coco/coco_train2014',
#           'datasets/data/coco/coco_val2014/'))
# parser.add_argument('--dset-rois', default=('files/proposal_files/coco_2014_train/rpn_proposals.pkl',
#                 'files/proposal_files/coco_2014_valminusminival/rpn_proposals.pkl'))
# parser.add_argument('--dset-ann', default=('datasets/data/coco/annotations/instances_train2014.json',
#                 'datasets/data/coco/annotations/instances_valminusminival2014.json'))
# parser.add_argument('--dset-path', default=('datasets/data/coco/coco_train2014',
#            ))
# parser.add_argument('--dset-rois', default=('files/proposal_files/coco_2014_train/rpn_proposals.pkl',
#                  ))
# parser.add_argument('--dset-ann', default=('datasets/data/coco/annotations/instances_train2014.json',
#                  ))
parser.add_argument('--dset-path', default=('datasets/data/coco/coco_val2014',
         ))
parser.add_argument('--dset-rois', default=('files/proposal_files/coco_2014_minival/rpn_proposals.pkl',
               ))
parser.add_argument('--dset-ann', default=('datasets/data/coco/annotations/instances_minival2014.json',
               ))
# DATALOADER

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
# SOLVER
parser.add_argument('--base-lr', default=0.01, type=float)
parser.add_argument('--lr-steps', default=[0, 240000, 320000])
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
# TRAINING
parser.add_argument('--max-iter', default=360000, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--start-iter', default=0, type=int, metavar='N',
                    help='manual iter number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint-period', default=20000, type=int)
parser.add_argument('--checkpoint-fn', default='files/results/fast.pth.tar')


def main():
    args = parser.parse_args()
    print(args)
    # for now, batch_size should match number of gpus
    assert(args.batch_size==torch.cuda.device_count())

    # create model
    model = detector(arch=args.cnn_arch,
                 base_cnn_pkl_file=args.cnn_pkl,
                 mapping_file=args.cnn_mapping,
                 output_prob=False,
                 return_rois=False,
                 return_img_features=False)
    model = model.cuda()

    # freeze part of the net
    stop_grad=['conv1','bn1','relu','maxpool','layer1']
    model_no_grad=torch.nn.Sequential(*[getattr(model.model,l) for l in stop_grad])
    for param in model_no_grad.parameters():
        param.requires_grad = False

    # define  optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    # create dataset
    train_dataset = CocoDataset(ann_file=args.dset_ann,
                          img_dir=args.dset_path,
                          proposal_file=args.dset_rois,
                          mode='train',
                          sample_transform=preprocess_sample(target_sizes=[800],
                                                             sample_proposals_for_training=True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=collate_custom)

    training_stats = TrainingStats(losses=['loss_cls','loss_bbox'],
                                   metrics=['accuracy_cls'],
                                   solver_max_iters=args.max_iter)

    iter = args.start_iter

    print('starting training')

    while iter<args.max_iter:
        for i, batch in enumerate(train_loader):

            if args.batch_size==1:
                batch = to_cuda_variable(batch,volatile=False)
            else:
                # when using multiple GPUs convert to cuda later in data_parallel and list_to_tensor
                batch = to_variable(batch,volatile=False)             
                

            # update lr
            lr = get_lr_at_iter(iter)
            adjust_learning_rate(optimizer, lr)

            # start measuring time
            training_stats.IterTic()

            # forward pass            
            if args.batch_size==1:
                cls_score,bbox_pred=model(batch['image'],batch['rois'])
                list_to_tensor = lambda x: x                
            else:
                cls_score,bbox_pred=data_parallel(model,(batch['image'],batch['rois'])) # run model distributed over gpus and concatenate outputs for all batch
                # convert gt data from lists to concatenated tensors
                list_to_tensor = lambda x: torch.cat(tuple([i.cuda() for i in x]),0)

            cls_labels = list_to_tensor(batch['labels_int32']).long()
            bbox_targets = list_to_tensor(batch['bbox_targets'])
            bbox_inside_weights = list_to_tensor(batch['bbox_inside_weights'])
            bbox_outside_weights = list_to_tensor(batch['bbox_outside_weights'])            
            
            # compute loss
            loss_cls=cross_entropy(cls_score,cls_labels)
            loss_bbox=smooth_L1(bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights)
                                  
            # compute classification accuracy (for stats reporting)
            acc = accuracy(cls_score,cls_labels)

            # get final loss
            loss = loss_cls + loss_bbox

            # update
            optimizer.zero_grad()
            loss.backward()
            # Without gradient clipping I get inf's and NaNs. 
            # it seems that in Caffe the SGD solver performs grad clipping by default. 
            # https://github.com/BVLC/caffe/blob/master/src/caffe/solvers/sgd_solver.cpp
            # it also seems that Matterport's Mask R-CNN required grad clipping as well 
            # (see README in https://github.com/matterport/Mask_RCNN)            
            # the value max_norm=35 was taken from here https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
            clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2) 
            optimizer.step()

            # stats
            training_stats.IterToc()
            
            training_stats.UpdateIterStats(losses_dict={'loss_cls': loss_cls.data.cpu().numpy().item(),
                                                        'loss_bbox': loss_bbox.data.cpu().numpy().item()},
                                           metrics_dict={'accuracy_cls':acc.data.cpu().numpy().item()})

            training_stats.LogIterStats(iter, lr)
            # save checkpoint
            if (iter+1)%args.checkpoint_period == 0:
                save_checkpoint({
                    'iter': iter,
                    'args': args,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, args.checkpoint_fn)

            if iter == args.start_iter + 20: # training_stats.LOG_PERIOD=20
                # Reset the iteration timer to remove outliers from the first few
                # SGD iterations
                training_stats.ResetIterTimer()

            # allow finishing in the middle of an epoch
            if iter>args.max_iter:
                break
            # advance iteration
            iter+=1
            #import pdb; pdb.set_trace()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)    

if __name__ == '__main__':
    main()
