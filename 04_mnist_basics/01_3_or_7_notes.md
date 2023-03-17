# Download and Visualize Data

We begin by downloading a sample of mnist's data. This data is just images of 3s and 7s. 
We also print out the path which the data is contained.

Note that the vast majority of the code is commented out, because those comments contain code for 
visualizing relevant functions. Just uncomment if you want to visualize any particular function 
being run.

```python
path = untar_data(URLs.MNIST_SAMPLE)
print(path.ls())
print((path/'train').ls())
```

Next, we'll create two lists: a sorted list that contains the path of the 3s, and another one for 
the 7s. Afterwards, we confirm by printing out the top 3 elements of the threes list.

```python
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
print(threes[:10])
```

Now, we'll visualize the data. We grab an element from `threes`, which is a path to an image, 
and crack it open for viewing.

```python
im3_path = threes[1]
im3 = Image.open(im3_path)
im3.show()
```

After looking at this image, we take at look at different representations of the image. 
Since an image is fundamentally a matrix of pixels, we can visualize the data as a numpy array, 
as well as a tensor.

```python
print(array(im3)[4:10,4:10]) # views rows from [4,10) and cols from [4,10)
print(tensor(im3)[4:10,4:10])
```

Then, we'll use a pandas dataframe to color code values using a gradient. 
Since each pixel is just a numerical value, we can use these values to make a greyscale image.

```python
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:26,4:22])
df_formatted = df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

To view this image, we generate an html file that draws it.

```python
with open('pixel_image.html', 'w') as image:
    df_formatted.to_html(image)
filename = 'pixel_image.html'
webbrowser.open_new_tab(filename)
```

![pixels](images/3_pixels.png)

# Attempting to Classify the Images

## Pixel Similarity

In this attempt, we will try to create a rough estimate for what a three and a seven should look 
like, based on the data that we were given. Then, we will compare our test data with that average 
three/seven, and see how close they are; the closer they are, the more likely it is that they are 
the three or seven.

We convert each image into a tensor, using Python List Comprehension 
(neat trick that has shorter syntax to create a new list based on the values of an existing list).

For this particular case: for every image in the list of sevens/threes, open the image and convert 
it into a tensor. This creates a list of tensors of our seven/three images. 
We print the lengths of the lists and view a table of the tensor to confirm that we're on the right 
path.

```python
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
print(len(three_tensors),len(seven_tensors))
show_image(three_tensors[1], cmap='binary')
plt.show()
```

Next, we will compute the average of each pixel by stacking the tensors into a rank 3 tensor 
(or a 3D tensor). Then we convert the tensors from `int` to `float`, so our averaging doesn't 
result in integer division. We divide by 255, so they're in the range of 
`[0,1]` instead of `[0,255]`. 

```python
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
```

By the way, here's some tensor jargon:
- `print(stacked_threes.shape)` are 6131 28x28 images stacked as a 3D tensor
- `print(len(stacked_threes.shape))` is the length of a tensor's shape. 
This is the same as the rank of the tensor
- `print(stacked_threes.ndim)` the rank of a tensor (ndim) = number of axes in a tensor 

Finally, we compute the "ideal" 3 and 7. This is done by computing the mean on the "horizontal" 
direction; this isn't a global mean of ALL values, that wouldn't tell us anything except how much 
the image is marked. In this case, our mean is made of 1 pixel per image; 
for example, we are taking the mean of all 6131 pixel values in `(0,0)`.

```python
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)

show_image(mean3, cmap='binary')
plt.show()
show_image(mean7, cmap='binary')
plt.show()
```

Now we test: we'll compare the distance between a sample 3 and the ideal 3 and 7. This can be done 
two ways - the L1 norm or the L2 norm. There is a whole discussion on when to use what norm 
(L1 is better for distances between zero and other small, non-zero elements since L2 grows slowly 
near the origin, which has implications for things like feature extraction, etc.). 
`dist7` should be larger than `dist3`, since the sample 3 should be "closer" to the ideal 3 than 
the ideal 7.

```python
# sample 3 and 7
a_3 = stacked_threes[1]
a_7 = stacked_sevens[1]

# distance b/t a datapoint and the ideals
dist3_abs = (a_3 - mean3).abs().mean()
dist3_sqr = ((a_3 - mean3)**2).mean().sqrt()

dist7_abs = (a_3 - mean7).abs().mean()
dist7_sqr = ((a_3 - mean7)**2).mean().sqrt()

print(dist3_abs, dist3_sqr)
print(dist7_abs, dist7_sqr)
```

PyTorch offers functions for calculating these as loss functions under `torch.nn.functional`, 
and the team recommends importing that as `F`. 
We don't have to since fastai makes it available under that name by default.

`print(F.l1_loss(a3.float(), mean7), F.mse_loss(a_3, mean7.sqrt()))`
- `F.l1_loss` obviously refers the L1 norm, or mean absolute error in this case. 
- `F.mse_loss` refers to the mean squared error, which we then follow up with a square root to get 
our root mean sqaured error (RMSE).

## Computing Metrics Using Broadcasting

At this point, we're looking to test our model against some data it's never seen. 
Of course, for the pixel similarity method, we dont' really need a validation set since there's no 
actual training happening. Still, we'll follow these practices (and we'll eventually need to do it 
with trained models in the future).

We'll start by going into our validation set folder, and creating a tensor out of each image. 
We cast the elements to float and divide by 255 so each element is in the range of [0,1]. 
The shape of each tensor reveals roughly 1000 28x28 pixel images per label.

```python
valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255

valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_3_tens.float()/255
print(valid_3_tens.shape, valid_7_tens.shape)
```

We will eventually need a function that we can call to measure the distance between the ideal image 
and our test image:

`def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))`

### Broadcasting

We'll come back to what the `(-1,-2)` tuple in `mean()` means after this. 
First, we should be concerned about how to get the distance between every element in the 
validation set and the ideal image. We could use a for loop, calculating the distance between 
each image in the set in our stacked tensor withe ideal image - or we could just pass in 
that stacked tensor:

```python
valid_3_dist = mnist_distance(valid_3_tens, mean3)
print(valid_3_dist, valid_3_dist.shape)
```

This prints out a tensor that is the exact size of the stacked tensor - no for loop required! 
How? PyTorch uses a method called broadcasting. When subtraction between two tensors of different 
ranks occurs, PyTorch automatically expands the tensor with the smaller rank to match the tensor 
of the larger rank - this makes tensor code easier to write.

![broadcasting](images/broadcasting.png)

Note that PyTorch isn't actually allocating memory to create a larger-ranked tensor from the 
smaller-ranked tensor; that's wholly unnecessary. The operation we defined above allows us to 
pretend that we're working with two tensors of the same rank without creating a new tensor of 
that larger rank.

### What the `mean()` Doing?

The `(-1,-2)` ordered pair is important. Note that eventually, we will be computing the distance 
between our entire validation set of images to the ideal image; that entails passing a 
rank-3 tensor. If we were to just compute the distance, `mean` would get the mean distances of 
all the images in the validation set, then calculate the mean of all those distances, 
returning one value. What we want is a vector of the distances between each image in the set 
and the ideal image.

To communicate that, `mean((-1,-2))` says to calculate the mean ranging over the values indexed 
by the last and second to last axes of the tensor. For our rank-3 tensor, that would be the 
horizontal and vertical dimensions, since it would be indexed as `(image number, horizontal pixel, 
vertical pixel)`, as given by `torch.shape`. So the tuple could be `(1,2)` or `(-1,-2)`; 
either way, we just need to tell the mean function to only get the mean out of each image. 
Here's an image to help visualize the function:

![mean_function](images/mean_function.png)

### Computing Accuracy

Continuing on, we want to determine if an image is a 3 or a 7; a simple way to implement that is 
to compare the distance our sample is from the ideal 3 and 7, and say whatever label that 
sample is closer to is the label our sample should be - and thanks to broadcasting, we can 
pass in the full validation set, not just one element of it:

```python
def is_3(x): return mnist_distance(x, mean3) < mnist_distance(x, mean7)
print(is_3(a_3), is_3(a_3).float())
print(is_3(valid_3_tens))
```

Now, all that remains is to calculate the accuracy of our simple model against the validation set:

```python
accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()
print(accuracy_3s, accuracy_7s, (accuracy_3s+accuracy_7s)/2)
```

Here we get accuracies in excess of 90%, so that's neat. But these are only two digits, and 
very distinct ones at that. Further on we'll investigate stochastic gradient descent, so 
our model can do some actual learning.

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
