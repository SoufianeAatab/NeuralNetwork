import numpy as np
import matplotlib.pyplot as plt

# (n_x, m)
# m number of examples
m = 10
x = np.zeros((2,m))
y = np.zeros((1,m))
w = np.array([[1.],[2.]])
b = 2

for i in range(10):
    a = np.random.randint(0,2)
    b = np.random.randint(0,2)
    y[0][i] = 1 if a==1 and b==1 else 0 # AND Logic Gate NN
    x[0][i] = a
    x[1][i] = b

# mean_squared_error
# Loss Function: L
# a = NN output, y = correct value
def cost(a, y):
    m = a.shape[1]
    cost = np.sum(np.power(a-y,2))
    return cost / m

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(x, w):
    # a = f(WT*x + b)
    return sigmoid(np.dot(w.T,x) + b)

def backward(x,w):
    m = x.shape[1]
    a = forward(x,w)
    # Loss function derivative: dL(a,y) / da = (a-y)
    # Sigmoid deriviative: da/dz = a * (1-a)
    # dL(a,y) / dz = a * (1-a) * (a-y)
    dz = a * (1-a) * (a-y)
    # dL(a,y) / dw1 = dz/dw1 * da/dz * dL(a,y) / da
    # z = w1*x1 + w2*x2
    # dz/dw1 = x1
    # dw1 = x1 * a * (1-a) * (a-y)
    # dw1 = xi * dz
    dw = np.dot(x,dz.T) / m
    db = np.sum(dz) / m
    return dw, db

a = forward(x,w)
lr = 0.1
costs = []
for i in range(1000):
    dw, db = backward(x,w)
    w -= lr * dw
    b -= lr * db
    a = forward(x,w)
    costs.append(cost(a,y))
plt.plot(list(range(1000)), costs, '-r')
plt.show()