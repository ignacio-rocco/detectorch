import torch
import collections
from torch.utils.data.dataloader import default_collate
import itertools

def collate_custom(batch,key=None):
    """ Custom collate function for the Dataset class
     * It doesn't convert numpy arrays to stacked-tensors, but rather combines them in a list
     * This is useful for processing annotations of different sizes
    """    
    
    # this case will occur in first pass, and will convert a
    # list of dictionaries (returned by the threads by sampling dataset[idx])
    # to a unified dictionary of collated values    
    if isinstance(batch[0], collections.Mapping):
        return {key: collate_custom([d[key] for d in batch],key) for key in batch[0]}
    # these cases will occur in recursion
    elif torch.is_tensor(batch[0]): # for tensors, use standrard collating function
        return default_collate(batch)
    elif isinstance(batch,list) and isinstance(batch[0],list): # lists of lists
        flattened_list  = list(itertools.chain(*batch))
        return flattened_list
    else: # for other types (i.e. lists), return as is
        return batch

