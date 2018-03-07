import os
import torch

def main():
    libpath=os.path.join(os.path.dirname(torch.__file__),'lib','include')
    print(libpath)


if __name__ == "__main__":
    main()

