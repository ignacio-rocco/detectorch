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

def to_cuda_variable(x,volatile=True):
    if isinstance(x,dict):
        return {key: to_cuda_variable(x[key],volatile=volatile) for key in x.keys()}
    if isinstance(x,list):
        return [to_cuda_variable(y) for y in x]
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.Tensor):
        if torch.__version__[:3]=="0.4" or volatile==False:
            return Variable(x.cuda())
        else:
            return Variable(x.cuda(),volatile=True)


def parse_th_to_caffe2(terms,i=0,parsed=''):
    # Convert PyTorch ResNet weight names to caffe2 weight names
    if i==0:
        if terms[i]=='conv1':
            parsed='conv1'
        elif terms[i]=='bn1':
            parsed='res_conv1'
        elif terms[i].startswith('layer'):
            parsed='res'+str(int(terms[i][-1])+1)
    else:
        if terms[i]=='weight' and (terms[i-1].startswith('conv') or terms[i-1]=='0'):
            parsed+='_w'
        elif terms[i]=='weight' and (terms[i-1].startswith('bn') or terms[i-1]=='1'):
            parsed+='_bn_s'
        elif terms[i]=='bias' and (terms[i-1].startswith('bn') or terms[i-1]=='1'):
            parsed+='_bn_b'
        elif terms[i-1].startswith('layer'):
            parsed+='_'+terms[i]
        elif terms[i].startswith('conv') or terms[i].startswith('bn'):
            parsed+='_branch2'+chr(96+int(terms[i][-1]))
        elif terms[i]=='downsample':
            parsed+='_branch1'
    # increase counter
    i+=1
    # do recursion
    if i==len(terms):
        return parsed
    return parse_th_to_caffe2(terms,i,parsed)
