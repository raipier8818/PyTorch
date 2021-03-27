import torch

#BroadCasting

m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3]) # [3] -> [3,3]

print(m1+m2)

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# [1, 2]
# ==> [[1, 2],
#      [1, 2]]
# [3]
# [4]
# ==> [[3, 3],
#      [4, 4]]



