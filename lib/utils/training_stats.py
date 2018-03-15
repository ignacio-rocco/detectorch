#!/usr/bin/env python2

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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import numpy as np

#from caffe2.python import utils as c2_py_utils
#from core.config import cfg
from utils.logging import log_json_stats
from utils.logging import SmoothedValue
from utils.timer import Timer


class TrainingStats(object):
    """Track vital training statistics."""

    def __init__(self, metrics, losses,
                 solver_max_iters):
        self.solver_max_iters = solver_max_iters
        # Window size for smoothing tracked values (with median filtering)
        self.win_sz = 20
        # Output logging period in SGD iterations
        self.log_period = 20
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(self.win_sz)
            for key in losses + metrics
        }
        self.losses_and_metrics = {
            key: 0
            for key in losses + metrics
        }
        self.smoothed_total_loss = SmoothedValue(self.win_sz)
        self.smoothed_mb_qsize = SmoothedValue(self.win_sz)
        self.iter_total_loss = np.nan
        self.iter_timer = Timer()
        self.metrics = metrics
        self.losses = losses

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self,losses_dict, metrics_dict):
        """Update tracked iteration statistics."""
        for k in self.losses_and_metrics.keys():
            if k in self.losses: # if loss
                self.losses_and_metrics[k] = losses_dict[k]
            else: # if metric
                self.losses_and_metrics[k] = metrics_dict[k]

        for k, v in self.smoothed_losses_and_metrics.items():
            v.AddValue(self.losses_and_metrics[k])
        #import pdb; pdb.set_trace()
        self.iter_total_loss = np.sum(
            np.array([self.losses_and_metrics[k] for k in self.losses])
        )
        self.smoothed_total_loss.AddValue(self.iter_total_loss)
        self.smoothed_mb_qsize.AddValue(
            #self.model.roi_data_loader._minibatch_queue.qsize()
            64
        )

    def LogIterStats(self, cur_iter, lr):
        """Log the tracked statistics."""
        if (cur_iter % self.log_period == 0 or
                cur_iter == self.solver_max_iters - 1):
            stats = self.GetStats(cur_iter, lr)
            log_json_stats(stats)

    def GetStats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (
            self.solver_max_iters - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        #mem_stats = c2_py_utils.GetGPUMemoryUsageStats()
        #mem_usage = np.max(mem_stats['max_by_gpu'][:cfg.NUM_GPUS])
        stats = dict(
            iter=cur_iter,
            lr="{:.6f}".format(float(lr)),
            time="{:.6f}".format(self.iter_timer.average_time),
            loss="{:.6f}".format(self.smoothed_total_loss.GetMedianValue()),
            eta=eta,
            #mb_qsize=int(np.round(self.smoothed_mb_qsize.GetMedianValue())),
            #mem=int(np.ceil(mem_usage / 1024 / 1024))
        )
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = "{:.6f}".format(v.GetMedianValue())
        return stats