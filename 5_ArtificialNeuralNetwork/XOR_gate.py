def AND_gate(x1, x2):
    w1=0.5
    w2=0.5
    b=-0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1

def NAND_gate(x1, x2):
    w1=-0.5
    w2=-0.5
    b=0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1

def OR_gate(x1, x2):
    w1=0.6
    w2=0.6
    b=-0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1

def XOR_gate(x1,x2):
    x_1 = NAND_gate(x1,x2)
    x_2 = OR_gate(x1,x2)
    w1 = 0.5
    w2 = 0.5
    b = -0.5
    result = w1*x_1 + w2*x_2 + b
    if result <= 0:
        return 0
    else :
        return 1


x_arr = [[0,0],[0,1],[1,0],[1,1]]

for x in x_arr:
    print(XOR_gate(x[0],x[1]))