# MNIST Loss Function

## Preparing the data and their labels

Recall that our data is currently formatted as a list of matrices, via `stacked_threes` and 
`stacked_sevens`. We need to convert this list of matrices into a list of vectors; this process 
of converting an `m x n` matrix to a `mn x 1` column vector is called **vectorization**, which 
stacks the columns of a matrix on top of each other. 
This makes for faster computation because matrix multiplication would require for-loops. 
But with two vectors, we can just compute the product between the vector of weights and the 
vector of inputs, which is a job that is incredibly easy to parallelize. 
[[1]](https://medium.com/@jwbtmf/vectorization-in-deep-learning-c47f0d171d0a)

More on vectorization:

>Basically, between python for loop and vectorized numpy arrays, the for loop is in raw python 
>that gets interpreted; loops assess the type of the operands at each iteration, which 
>introduces a severe amount of computational overhead. Meanwhile, numpy uses BLAS (basic linear 
>algebra subprograms) written in c/fortran that has incredibly optimized and low-level linear 
>algebra operations. Finally, vector operations can also be done in parallel using the SIMD 
>paradigm, further introducing performance gains unaccessible to raw python for loops.
>[[2]](https://www.r-bloggers.com/2018/05/machine-learning-explained-vectorization-and-matrix-operations/) 
>[[3]](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)

```python
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
```

Continuing on, we collect our training data under `train_x`, where we create a tensor with 
`stacked_threes` concatenated with `stacked_sevens`. The `view(-1, 28*28)` function 
reshapes/vectorizes this tensor, with `-1` denoting "make this axis as large as necessary to fit 
the data". The result is a list of vectors; in this case, a tensor of the shape `(12396, 784)`, 
denoting 12396 images, each as a vector of 784 pixels.

Of course, we also need to label the data; we'll choose `1` for threes and `0` for sevens.

```python
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
```

Decoding this python nonsense, `[1]*len(threes)` generates a list of size `len(threes)` whose 
elements are initialized to `1`. The addition of `[0]*len(sevens)` concatenates that list with a 
list of size `len(sevens)` whose elements are initialized to `0`. The next function, 
`unsqueeze(1)`, adds a dimension of size 1 at index 1, converting our list of 12396 into a 
proper tensor of shape `[12396, 1]`, matching the rank of our training data.

```python
print(train_x.shape, train_y.shape)
```

`torch.Size([12396, 784]) torch.Size([12396, 1])`

We'll match the training data with the training labels, by forming a pair between each image 
and label in the form `(image, label)`, which can be done using the `zip()` function. We then 
store each of these pairs into a list called `dset` using `list()`. 
Recall that each image was vectorized into vectors of size 784, and that a label is either 1 or 0. 
So each pair will be a rank 1 tensor of size 784 of values from 0 to 1, and a rank 0 tensor of 
value either 0 or 1.

```python
dset = list(zip(train_x, train_y))
x,y = dset[0]
print(x.shape, y)
```

`torch.Size([784]) tensor([1])`

Finally, we'll prepare the validation set and their labels in the same manner as above.

```python
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))
```

## SGD

Now we'll begin the stochastic gradient descent process. First, we'll need to randomly 
initialize our weights for every pixel.

### init

Recall that on top of initializing random weights, we also need to inclue a bias 
`y = mx + b` since `weights * pixels` alone is not flexible enough, since it will always be 
0 if the pixels are 0. Together, the *weights* and the *bias* make up the *parameters*.

```python
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
weights = init_params((28*28, 1))
bias = init_params(1)
```

### predict

First, let's see a prediction for a single image in our training set.

```python
print((train_x[0]*weights.T).sum() + bias)
```

`tensor([-6.2330], grad_fn=<AddBackward0>)`

Here `weights.T` is just the transpose of `weights`, converting the column vector into a row 
vector. That way, we're doing element-wise vector multiplication.

Now, there is a temptation to use for loops to calculate the predictions for all the images 
in the training data. However, we're talking about python here - this is a huge mistake. We 
want to convert this into a matrix multiplication problem and let BLAS do the work (see the 
previous discussion above on vectorization).

```python
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
print(preds)
```

```
tensor([[ -6.2330],
        [-10.6388],
        [-20.8865],
        ...,
        [-15.9176],
        [ -1.6866],
        [-11.3568]], grad_fn=<AddBackward0>)

```

This equation, `batch @ weights + bias`, is one of the two fundamental equations of any neural 
network, so it's worth understanding to the fullest.

Using broadcasting (no loops), we can also check the accuracy of our predictions:

```python
corrects = (preds>0.0).float() == train_y
```

That is, get our tensor of predictions; for every element greater than 0, mark it as true; else 
mark it as false. Then convert those T/F values into floats, or 1.0/0.0, and mark it as true 
if it matches the corresponding element value in `train_y`. The result is a tensor of T/F 
values, where true means the prediction for that image was correct, false if otherwise. In other 
words, a tensor of prediction accuracies.

```python
print(corrects)
```

```
tensor([[False],
        [False],
        [False],
        ...,
        [ True],
        [ True],
        [ True]])
```

We then get the tensor of prediction accuracies, convert their elements to float, get the mean of 
the tensor, giving us a scalar. We use `item()` to convert a scalar tensor into a standard Python 
number - in this case, the accuracy of the model's predictions as a whole.

```python
print(corrects.float().mean().item())
```

```
0.5379961133003235
```

Here, we'll see what changes to one weight does to the accuracy:

```python
with torch.no_grad(): weights[0] *= 1.0001
preds = linear1(train_x)
print(((preds>0.0).float() == train_y).float().mean().item())
```

```
0.5379961133003235
```

>Note that we don't need to recalculate gradients here since we're just testing a weight change, 
>not doing the process of getting gradients to find out where our weights should be moved.

### loss (that gradient is calculated on)

Now that we can make predictions, we need to create a loss function that measures how good 
our weights are at producing results, that we can then calculate gradients on.

We already have accuracy as our metric, and it would be reasonable to assume it would make for 
a good loss function. However, the issue is accuracy is binary - the model either gets the 
solution or it doesn't - there is no concept of "getting closer" to the solution, but 
falling short, while recognizing we're going in the right direction. It's like if a professor 
graded your test by marking the wrong answers, without giving explanations for why you got 
something wrong. In the context of our loss function, that would essentially mean that 
such change to weights don't really change the loss function's curve; that is, our 
gradient is 0 almost everywhere. Much like how you can't learn from a test that is marked without 
explanations, the model will not be able to learn if different weight values don't change the 
loss curve.

>Mathematically the loss function with the accuracy metric would look like a step function; the 
>derivative at any point is either 0 or infinity. If the derivative is 0, then we don't know if 
>we should increase or decrease the weights to get a minimum.

So instead of a binary yes/no prediction, we'll turn to probabilities to craft our loss 
function. Fundamentally, a loss function is meant to measure the "distance" between a 
prediction and reality. Instead of 0 and 1 prediction for our image, what if we instead 
had a confidence interval? For example, instead of guessing if an image is a 3, the model 
instead gave a number between 0 and 1 that represents *confidence* that the image is a 3. 
That is, a model giving 0.75 for an image means it's more confident it's a 3 than an image 
marked as 0.25.

Let's take a shot at defining a loss function of this kind:

```python
trgts = tensor([1,0,1])
prds = tensor([0.9,0.4,0.2])

def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

Here `trgts` represents if the image is actually a 3 or a 7. Then `prds` would be the alleged 
predictions on what the images are in terms of how confident it is. So for the first image the 
model is 90% confident it's a 3, but for the 3rd image it's only 20% confident.

As for `torch.where(a,b,c)`, this function is the same as Python's list comprehension, 
except done on tensors using C/CUDA speed. That is, this is the same as the piecewise function:  

`myList[i] = b[i] if a[i]==True else c[i] for i in range(len(a))`. 
[[4]](https://pytorch.org/docs/stable/generated/torch.where.html)

In the context of our loss function, we're telling PyTorch to make a tensor whose elements are 
`1-prds[i]` if `trgts[i]==1`, else use `prds[i]`. We write our tensor like this because we 
want to have each element be the distance from 0 if it should be 0, and the distance from 1 if 
it should be 1, so now we achieved our goal of getting the elements of the tensor to show the 
distance between the prediction and the target, not distance between only 0 or only 1.

```python
print(torch.where(trgts==1, 1-prds, prds))
```
`tensor([0.1000, 0.4000, 0.8000])`

And of course, since the loss function is a multivariate function with one output, we return 
the mean of these distances as a rank 0 tensor (scalar). We can also see that by moving the 
third image prediction from 0.2->0.8 (closer to the target 1), that loss declines, as expected.

```python
print(mnist_loss(prds,trgts))
print(mnist_loss(tensor([0.9,0.4,0.8]),trgts))
```

```
tensor(0.4333)
tensor(0.2333)
```

#### sigmoid

While using confidence is nice, our current setup assumes that the predictions are between 
0 and 1, which definitely not the case as you can see from the previous predictions that we 
made. Thus we need to fit our predictions in such a manner that matches our expectations.

To do this, we turn to the sigmoid function:

```python
# actual function
def sigmoid(x): return 1/(1+torch.exp(-x))

# using torch's implementation
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
plt.show()
```
![sigmoid](images/sigmoid.png)

Some important characteristics of this function includes: 
- Output is in the range of 0 and 1
- Output *saturates* when the input is very positive or negative; meaning at more extreme values 
the function is flat and insensitive to change

Applied to our loss function, we have:

```python
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

To recap, we unearthed the difference between a good metric and a good loss function. A good 
metric is human-readable and shows us how the model is performing, and a good loss function 
is a measure of how well the model is performing + meaningful derivatives. In our case, metrics 
as a loss function hid "progress" and can't help our model learn. Put simply, loss is for 
learning and metrics are for judging.

### step (optimization)

Recall weights and input plus bias make the prediction, and the loss function is a measure 
of how far the prediction is from reality. Our loss function can be generated using one input, 
or perhaps loss can be calculated for the entire dataset and a point on the curve would be 
the average of all those losses. However, both methods are dangerous.

First, using only one data point to guide your loss function leaves you susceptible to 
huge error - what if the data point you happen to choose is some terrible outlier? You 
don't want to use these outliers to teach your model.

Second, calculating the average loss over an entire data set is unteneable for large data 
sets; it would simply take too long.

So if 1 data point is too imprecise and all data points is too time-consuming, what is the 
middle ground? Mini-batches provide the solution.

Mini-batches are a sample of the data set, and we take calculate the average loss over this 
sample. The larger the batch size, the more accurate the loss; but the more computation time 
is required. 

Your choice in hardware also determines how large your batch size can/needs to be. Ideally 
you can choose a batch size that makes full use of your GPU, but doesn't exceed its memory 
limitations.

To select data for our mini-batches, we want random samples - so we use `DataLoaders` to 
do shuffling and mini-batch collation, like so: 

```python
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
print(list(dl))
```

`[tensor([ 3, 12,  8, 10,  2]), tensor([ 9,  4,  7, 14,  5]), tensor([ 1, 13,  0,  6, 11])]`

Here we generated a list of numbers from 0 to 14, defined a batch size of 5 
(so 3 batches of size 5), and shuffled the numbers.

Another thing to mention is that we're not just using normal Python collections, but a 
collection of independent variables (inputs) and dependent variables (targets) - so we'll 
need the ability to have a collection of tuples containing our inputs and targets, managed 
by PyTorch's `DataSet`. This `DataSet` object is then passed into a `DataLoader`; here's a 
simple example with inputs as integers and targets as characters.

```python
ds = L(enumerate(string.ascii_lowercase))
dl = DataLoader(ds, batch_size=6, shuffle=True)
print(ds)
print(list(dl))
```

```
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i'), (9, 'j'), (10, 'k'), (11, 'l'), (12, 'm'), (13, 'n'), (14, 'o'), (15, 'p'), (16, 'q'), (17, 'r'), (18, 's'), (19, 't'), (20, 'u'), (21, 'v'), (22, 'w'), (23, 'x'), (24, 'y'), (25, 'z')]

[(tensor([17, 18, 10, 22,  8, 14]), ('r', 's', 'k', 'w', 'i', 'o')), (tensor([20, 15,  9, 13, 21, 12]), ('u', 'p', 'j', 'n', 'v', 'm')), (tensor([ 7, 25,  6,  5, 11, 23]), ('h', 'z', 'g', 'f', 'l', 'x')), (tensor([ 1,  3,  0, 24, 19, 16]), ('b', 'd', 'a', 'y', 't', 'q')), (tensor([2, 4]), ('c', 'e'))]
```

# Fully implementing gradient descent

## What happens at each epoch

The goal of this section is to implement the gradient descent process; or defining what 
happens at each epoch. We want something like this: 

```python
for x,y in dl:
  pred = model(x)                     # predict
  loss = loss_func(pred, y)           # calc loss
  loss.backward()                     # calc gradients of loss func
  parameters -= parameters.grad * lr  # update parameters, given some learning rate
```

To get here, we'll start by initializing our parameters:

```python
weights = init_params((28*28,1))
bias = init_params(1)
```

Then, we'll collect our training and validation data into DataLoader batches:

```python
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

xb,yb = first(dl) 
print(xb.shape, yb.shape)
```

`torch.Size([256, 784]) torch.Size([256, 1])`

Here we pass in the training dataset, and tell `DataLoaders` to put the images in batches of 
size 256. Recall that `dset` is a list of pairs of vectorized images and labels. By 
using `fastcore.basics.first()` we get the first element in our `DataLoaders` object, and 
find that each element is a pair with 256 vectorized images and 256 labels.

Now, we'll do a mini-batch size of 4 to test:

```python
batch = train_x[:4]
print(batch.shape)

preds = linear1(batch)
print(preds)

loss = mnist_loss(preds, train_y[:4])
print(loss)

loss.backward()
print(weights.grad.shape, weights.grad.mean(), bias.grad)
```

```
torch.Size([4, 784])
tensor([[14.0882],
        [13.9915],
        [16.0442],
        [17.7304]], grad_fn=<AddBackward0>)
tensor(4.1723e-07, grad_fn=<MeanBackward0>)
torch.Size([784, 1]) tensor(-5.9512e-08) tensor([-4.1723e-07])
```

Here, we:
- Made a batch size of 4 with the training data
- Performed predictions on this batch with our randomly initialized weights and bias
- Calculated the loss of these predictions
- Calculated the gradients of our loss function w.r.t. the parameters

Since we'll be doing this once per epoch across many epochs, we'll make a function out of this 
process:

```python
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
```

Testing it, we notice that if you call `calc_grad()` more than once, the gradients change 
despite the function and the inputs not changing at all.

```python
calc_grad(xb, yb, linear1)
print(weights.grad.mean(), bias.grad)

calc_grad(xb, yb, linear1)
print(weights.grad.mean(), bias.grad)
```

```
tensor(-0.0035) tensor([-0.0273])
tensor(-0.0069) tensor([-0.0546])
```

This is because `loss.backward()` adds the gradients of loss to the currently stored gradients. 
So for subsequent calls, we'll have to make sure to reset the stored gradient values using 
these lines:

```python
weights.grad.zero_()
bias.grad.zero_()
```

To round out our basic training loop for an epoch, we'll have to update the weights and 
biases based on the gradient and learning and ensure we reset the gradients right after: 

```python
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()
```

That is for each epoch, we:
- For each item in the batch, calculate the gradients (predict, loss, gradient)
- Then for every parameter in that item, update the weights (based on gradient and learning rate), 
then reset the gradients

Now we'll take on calculating accuracy and validating each epoch. On our mini-batch, it would 
be:

```python
print((preds>0.0).float() == train_y[:4])
```

```
tensor([[True],
        [True],
        [True],
        [True]])
```

As a function, we want to transform the predictions to conform to a sigmoid function, 
check the correct predictions, then return the mean: 

```python
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

print(batch_accuracy(linear1(batch), train_y[:4]))
```

`tensor(1.)`

Combining the accuracies of the batches together, we get this function:

```python
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

print(validate_epoch(linear1))
```

`0.9697`

That is, make a list of tensors of batch accuracies for every batch. Get the list, stack all 
the tensors into one tensor of accuracies, get the mean of all the accuracies, cast it into 
a Python number, and round it to 4 decimal places.

Finally, we'll train for a few epochs and see what happens:

```python
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
print(validate_epoch(linear1))

for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
print()
```

```
0.7344
0.8687 0.9229 0.9429 0.9526 0.9595 0.9678 0.9707 0.9761 0.979 0.9805 0.9814 0.9819 0.9854 0.9854 0.9854 0.9858 0.9868 0.9868 0.9868 0.9873 
```

So we got our expected result - as we train our model, the accuracy of it improves!

## Creating an optimizer

This process is both fundamental and foundational - so it's no surprise that PyTorch already 
has some helpers for us to use to simplify our code.

### Using PyTorch

To start off with, we can replace `init_params()` and `linear1()` with PyTorch's `nn.Linear` 
module; we just need to pass in the shape of the parameters and the bias:

```python
linear_model = nn.Linear(28*28, 1)
w,b = linear_model.parameters()
print(w.shape,b.shape)
```

`torch.Size([1, 784]) torch.Size([1])`

With `nn.Linear` we can make a basic optimizer, responsible for the step and zeroing gradients: 

```python
class BasicOptim:
    def __init__(self, params, lr): self.params,self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```

Then we can simplify our `train_epoch()` function using the class we defined: 

```python
opt = BasicOptim(linear_model.parameters(), lr)

def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

print(validate_epoch(linear_model))
```

`0.2129`

In this case, `validate_epoch()` doesn't need to be changed at all.

We'll then put our training loop in a function: 

```python
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
    print()

train_model(linear_model, 20)
```

`0.4932 0.8071 0.853 0.917 0.937 0.9502 0.9595 0.9668 0.9688 0.9712 0.9731 0.9746 0.9771 0.978 0.979 0.98 0.9805 0.981 0.981 0.9819`

### Using fastai

Here we can also use fastai, which provides an SGD class that does what our `BasicOptim` class 
does:

```python
linear_model = nn.Linear(28*28, 1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```

`0.4932 0.875 0.8291 0.9121 0.9355 0.9492 0.9585 0.9663 0.9692 0.9707 0.9722 0.9746 0.9771 0.978 0.979 0.979 0.9805 0.981 0.9814 0.9814`

In fact, if we have our: 
- Training and validation set in a DataLoaders object
- Choice of optimizer from fastai
- Loss function defined
- Metrics function defined
- Learning rate defined

Then we can use fastai's `Learner` class to do it all; using `Learner.fit()` to replace 
`train_model()`

```python
dls = DataLoaders(dl, valid_dl)
learn = Learner(dls, nn.Linear(28*28, 1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(10, lr=lr)
```

```
epoch     train_loss  valid_loss  batch_accuracy  time
0         0.637435    0.278713    0.500000        00:00
1         0.351899    0.532999    0.683663        00:00
2         0.136168    0.369614    0.858911        00:00
3         0.063544    0.307757    0.919307        00:00
4         0.036521    0.279117    0.937624        00:00
5         0.025754    0.261695    0.951980        00:00
6         0.021192    0.249678    0.960396        00:00
7         0.019066    0.240751    0.966832        00:00
8         0.017919    0.233703    0.968812        00:00
9         0.017185    0.227875    0.971287        00:00
```


