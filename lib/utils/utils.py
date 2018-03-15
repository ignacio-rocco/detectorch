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

def to_variable(x,volatile=True):
    if isinstance(x,dict):
        return {key: to_variable(x[key],volatile=volatile) for key in x.keys()}
    if isinstance(x,list):
        return [to_variable(y,volatile=volatile) for y in x]
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.Tensor):
        if torch.__version__[:3]=="0.4" or volatile==False:
            return Variable(x)
        else:
            return Variable(x,volatile=True)


def to_cuda_variable(x,volatile=True,gpu=None):
    if isinstance(x,dict):
        return {key: to_cuda_variable(x[key],volatile=volatile,gpu=gpu) for key in x.keys()}
    if isinstance(x,list):
        return [to_cuda_variable(y,volatile=volatile,gpu=gpu) for y in x]
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.Tensor):
        if torch.__version__[:3]=="0.4" or volatile==False:
            return Variable(x.cuda(gpu))
        else:
            return Variable(x.cuda(gpu),volatile=True)



def isnan(x):
    return x != x

def nanbreak(grad):
    nancount = torch.sum(isnan(grad).float()).cpu().data.numpy().item()
    if nancount>0:
        import pdb; pdb.set_trace()
    return None

def infbreak(grad):
    max = torch.max(grad).cpu().data.numpy().item()
    if max>1e10:
        import pdb; pdb.set_trace()
    return None

def printmax(grad):
    max = torch.max(grad).cpu().data.numpy().item()
    print(max)
    return None
