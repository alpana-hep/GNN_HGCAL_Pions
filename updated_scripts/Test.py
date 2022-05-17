import torch
suma = torch.load("./checkpoints/model_checkpoint_DataParallel_62405_7aad117da0_root.best.pth.tar",map_location=torch.device('cpu'))
print(suma.keys())
print(suma.model)
print(model.load_state_dict(suma))
