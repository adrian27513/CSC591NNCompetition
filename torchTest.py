import torch.nn.functional as F
import torch.nn as nn
import torch

loss = nn.CrossEntropyLoss()
input = F.softmax(torch.tensor([[1,0,0,0],[1,0,0,0],[1,0,0,0]]).to(torch.float32), dim=1)
# print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
target = torch.tensor([[1,0,0,0],[1,0,0,0],[1,0,0,0]]).to(torch.float32)
# print(target)
output = loss(input, target)
# print(output)