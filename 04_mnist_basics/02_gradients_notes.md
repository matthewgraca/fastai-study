# Brief Look at Stochastic Gradient Descent (SGD) 

## Loss as a Function w.r.t. Weight Parameters 

Whenever we train a model, we want to know what kind of weights we want to assign each variable, such that loss is minimized. We can actually plot a function that describes this behavior - in a simple example, take `f(x) = x**2`. Suppose `f(x)` is loss, given a particular weight parameter `x`. In this case, we find that loss is at its lowest when the parameter `x = 0`.

SGD is the process in which we find these minimums and use them for adjusting our weights. Since the goal of our model is to adjust weights such that loss is minimized, and we can turn this into a function, we turn to calculus and use gradients to observe where our parameters should be "going".

For example, if we arbitrarily begin at `x = -1.5`, we should increase `x` to minimize loss. Why? Because the function is decreasing at `x = -1.5`. So we set `x = -1`, find out that the function is still decreasing, then `x= -0.75`, and continue on until we hit the function's minimum at `x = 0`.

```
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red')
plt.show()
```

In actual models there will be many weights, so in reality we're dealing with multivariable functions - exactly how the gradient is calculated for those kind of functions is a calculus question (Jacobians and Chain Rule). For now, just know that we can calculate the gradient of a particular function at a particular value using a few special functions.

## Calculating the Gradient (Backpropagation)

First we mark a tensor, essentially saying that this set of variables requires gradient calculations. This is necessary because there are instances where you will have a different set of variables that are inputs, but whose derivatives are not required. 

```
xt = tensor(3.).requires_grad_()
print(xt)
```

`tensor(3., requires_grad=True)`

We pass that marked tensor containing the value `x` we want to calculate the gradient at into the function. Right now, no gradients are being returned; the tensor that is returned contains the value at `f(x)`, and is marked with the gradient function that will be used when asked to calculate the gradient at that `x` value.

```
yt = f(xt)
print(yt)
```

`tensor(9., grad_fn=<PowBackward0>)`

Now, we'll use a vector argument instead of a scalar argument.

```
xt = tensor([3., 4., 10.]).requires_grad_()
yt = f(xt)
yt.backward()
print(xt.grad)
```

`tensor([ 6.,  8., 20.])`

Notice a few changes; first, that `f(x)` has been modified to return a scalar value instead of a vector - which has some implications.

```
def f(x): return (x**2).sum()
print(yt)
```

`tensor([  9.,  16., 100.], grad_fn=<PowBackward0>)`

This is because `backward()` by default only works on scalar outputs - this is because the expected out is a scalar loss function. This makes more sense if you understand the chain rule and the computational graph implementation, so that's beyond the scope of this simple explanation.

Note that all we've done is calculate the gradient - this action in itself isn't SGD, but backpropagation. Backpropagation and calculating the gradient are synomyms - SGD refers to the entire learning process (that requires knowing the gradient of our loss function).

At this point, the gradient tells us "where to go", but not "how far we should move". What is the best rate to approach the minimum at? Learning rate gives us an answer.
