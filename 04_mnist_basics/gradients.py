from fastbook import * 
import matplotlib.pyplot as plt

def f(x): return (x**2).sum()

'''
# suppose f(x) is the loss of a given weight parameter x
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red')
plt.show()
'''

# calc gradients with scalar arg
xt = tensor(3.).requires_grad_()
print(xt)

yt = f(xt)
print(yt)

yt.backward()
print(xt.grad)

# calc gradients with vector arg
xt = tensor([3., 4., 10.]).requires_grad_()
print(xt)

yt = f(xt)
print(yt)

yt.backward()
print(xt.grad)
