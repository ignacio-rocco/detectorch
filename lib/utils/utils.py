import torch
from torch.autograd import Variable

def normalize_axis(x,L):
    return (x-1-(L-1)/2)*2/(L-1)

def unnormalize_axis(x,L):
    return x*(L-1)/2+1+(L-1)/2

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))

def create_file_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def to_cuda(x):
    if isinstance(x,dict):
        return {key: to_cuda(x[key]) for key in x.keys()}
    if isinstance(x,list):
        return [y.cuda() for y in x]
    return x.cuda()

def to_cuda_variable(x):
    if isinstance(x,dict):
        return {key: to_cuda_variable(x[key]) for key in x.keys()}
    if isinstance(x,list):
        return [to_cuda_variable(y) for y in x]
    if torch.__version__[:3]=="0.4":
        return Variable(x.cuda())
    else:
        return Variable(x.cuda(),volatile=True)

