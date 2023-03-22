This section of notes will be covering the data, as well as preliminary attempts 
to classify it.

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


