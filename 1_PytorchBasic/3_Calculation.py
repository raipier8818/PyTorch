import torch

# Matrix Multiplication

m1 = torch.FloatTensor([[1, 2],
                        [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1


# Multiplication (Broadcasting -> element-wise) == *

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))


# Mean

t = torch.FloatTensor([1,2])
print(t.mean())

t = torch.FloatTensor([[1,2],[3,4]])
print(t.mean())

print(t.mean(0))
print(t.mean(1))


# Sum

t = torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.sum())
print(t.sum(0))
print(t.sum(1))
print(t.sum(-1))


# Max & ArgMax

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max()) # Returns one value: max
print(t.max(dim=0)) # Returns two values: max and argmax

print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])