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

`tensor([13.4192], grad_fn=<AddBackward0>)`

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
tensor([[13.4192],
        [ 8.3099],
        [18.2619],
        ...,
        [13.6045],
        [ 9.9905],
        [17.1171]], grad_fn=<AddBackward0>)
```

This equation, `batch @ weights + bias`, is one of the two fundamental equations of any neural 
network, so it's worth understanding to the fullest.

Using broadcasting, we can also check the accuracy of our predictions:

```python
corrects = (preds>0.0).float() == train_y
print(corrects)
print(corrects.float().mean().item())
```


