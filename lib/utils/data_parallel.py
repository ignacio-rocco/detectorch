import operator
import torch
import warnings
from torch.nn import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply

class DataParallel(torch.nn.DataParallel):
    def __init__(self, *args, **kwargs):
            super(MyDataParallel, self).__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids): # scatter a list of len N into N gpus
        return scatter_lists(inputs, kwargs, device_ids)

def scatter_lists(inputs, kwargs,device_ids):
        n_inputs = len(inputs)
        n_devices = len(device_ids)
        for i in range(n_inputs):
            assert(len(inputs[i])==n_devices)
        inputs=tuple([tuple([inputs[i][j].cuda(device_ids[j]) for i in range(n_inputs)]) for j in range(n_devices)])
        return inputs,kwargs


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None, dont_scatter=False, dont_gather=False):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    #print('getting device_ids')
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    #print(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if dont_scatter==False:
        do_scatter_lists=isinstance(inputs[0],list)
        if do_scatter_lists:
            inputs, module_kwargs = scatter_lists(inputs, module_kwargs, device_ids)
        else:
            inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    #print('getting used device_ids')
    used_device_ids = device_ids[:len(inputs)]
    #print(used_device_ids)
    #print('making model replicas')
    replicas = replicate(module, used_device_ids)
    #print('applying model')
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    if dont_gather:
        return tuple([[out[i] for out in outputs] for i in range(len(outputs[0]))])
    #print('gathering result')
    return gather(outputs, output_device, dim)