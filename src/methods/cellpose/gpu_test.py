import torch
import sys

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    sys.exit()
