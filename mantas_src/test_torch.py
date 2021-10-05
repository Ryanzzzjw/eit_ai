import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.current_device())
