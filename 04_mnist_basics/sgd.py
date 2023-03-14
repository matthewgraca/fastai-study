# end to end SGD example
# goal: examine how finding a minimum of a loss function can be used to train a model to fit data better
from fastai.vision.all import *
import matplotlib.pyplot as plt

# suppose we are trying to model velocity of a rollercoaster, given our recorded data points
time = torch.arange(0,20).float();
print(time)

# model speed off of a quadratic equation, with some added noise
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed)
plt.show()

# we assume the function is some quadratic; we distinguish b/t the input and the parameters
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c

# in order to find the best set of params that fits the true function, we need a loss function that tells us how good our guess is
# use good old mse to measure distance b/t our predicted points and actual target points
def mse(preds, targets): return ((preds-targets)**2).mean()

# draws our preds and target function
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)

# step 1: random init
# generate 3 random nums as a tensor, then mark to track gradients
params = torch.randn(3).requires_grad_()

# step 2 calc preds
preds = f(time, params)
show_preds(preds)
plt.show()

# step 3 calc loss
loss = mse(preds, speed)
print(loss)

# step 4 calc gradients
loss.backward()
print(params.grad)
print(params.grad * 1e-5)
print(params)

# step 5 step the weights
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None
