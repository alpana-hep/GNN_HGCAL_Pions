import torch
import numpy as np
import pickle

a=torch.load("./checkpoints/model_checkpoint_DynamicReductionNetwork_62405_3996cea28e_asirohi.best.pth.tar", map_location=torch.device('cpu'))
print(a['model'])


