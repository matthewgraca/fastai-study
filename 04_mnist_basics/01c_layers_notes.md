# Adding Nonlinear Layers

## Brief Discussion of Nonlinear Layers in a Neural Net

Recall: what is a model? A model is a fundamentally a function. We take inputs and match 
them to desired outputs. In the same way, "reality" is also a function where we map inputs 
to their true outputs. The goal of the model is to get as close to this "reality" function 
as possible.

> It is best to think of feedforward networks as **function approximation machines** that are 
> **designed to achieve statistical generalization**, occasionally drawing some insights from 
> what we know about the brain, rather than as models of brain function. 
>
> "Deep Learning" by Goodfellow, Bengio, and Courville

Our current model has one linear layer - this means that we can only approximate linear 
functions - but reality is rarely simple to the point where such a function is sufficient. 
As a result, we need a model that can approximate nonlinear functions as well - another 
layer that can add curvature to our function.

**But why not linear?** Surely a composition of linear functions can give better approximations? 
Unfortunately, that is not the case. The reason why is because the composition of any two 
linear functions is simply another linear function; that is, *composition preserves linearity*. 
[[5]](https://www.statlect.com/matrix-algebra/composition-of-linear-maps) 
No matter how complicated the function, with any arbitrary input, we will only have a linear 
function.

In our model, we have various layers that tune our function so it can best match reality - 
and we've shown that linear functions are just too limited. In fact, linear functions are 
so limited it can't even approximate XOR. [[6]](https://stats.stackexchange.com/a/366131) 
Indeed, the *Universal Approximation Theorem* tells us that any continuous function f can be 
modeled with a neural network with just one hidden layer and a sufficient amount of units. 
But another condition of this theorem is that this hidden layer must be nonlinear. 
[[7]](https://towardsdatascience.com/if-rectified-linear-units-are-linear-how-do-they-add-nonlinearity-40247d3e4792) 
That is, if we want to model "reality", we're going to have to have a model that contains a 
nonlinear hidden layer. 

The function in this hidden layer is called an *activation function*. There are plenty we can 
choose from: the familiar sigmoid, to the exotic tanh. However, there is another function 
that is kind of linear, kind of not, but still fulfills the theorem: the Rectified Linear 
Unit, or ReLU.

```math
\begin{equation}
f(x)=x^{+}=max(0,x)=
  \begin{cases}
    x & \text{if } x > 0 \text{,}\\
    0 & \text{otherwise.}
  \end{cases}
\end{equation}
```

```math
\begin{equation}
f'(x)=
  \begin{cases}
    1 & \text{if } x > 0 \text{,}\\
    0 & \text{if } x < 0 \text{.}
  \end{cases}
\end{equation}
```

![relu_plot](images/relu.png)

If you look at a plot of the ReLU function, it doesn't look like much - flat for negative 
inputs, and linear for positive inputs. It seems like this isn't complicated enough to 
model the complexities of reality. However, the power of ReLU is not what it can do alone, 
but what it can represent as many. Much like how a binary value does nothing alone, many together 
serve as the fundamental building blocks of computing. *ReLU is that fundamental building 
block that, given enough units, can approximate any function*.

But perhaps we put the cart before the horse; we know ReLU is good, but **why do we choose it over 
other nonlinear functions** like sigmoid or tanh? Because it optimizes faster. Generally speaking, 
nonlinear functions are more sophisticated, *including their gradient calculations*. This is the 
grand advantage of ReLU; it's "curvy" enough to approximate nonlinear functions (given arbitrarily 
many), but linear enough to make gradient calculations super simple. The time save is 
dramatically large, that this function is worth it. 
[[8]](https://datascience.stackexchange.com/questions/37079/how-can-relu-ever-fit-the-curve-of-x%c2%b2/37080#37080) 
What you can do with ReLU in one hour would take tanh one month. 

## Neurons Per Layer

How many neurons should there be per layer? What does having more or less neurons do? How is 
work divided among these neurons?

Essentially, the more features and the more complex the training data is, the more neurons 
you need to parse those complexities. The learning algorithm will decide how to use those 
layers; they are not explicitly told by the training data what they should be, or what 
each layer should do. The learning algorithm is responsible for deciding how to use those 
layers to best approximate the "real" function.

To know how many neurons you need, you have to know about how your data is defined. For 
example, if you have a classification model that draws a boundary between two distinct 
groups of data, you would need only two neurons in your hidden layer to complete the 
job. But as the number of groups increase and intermingle, you may need more neurons 
and more layers, like in this example. 
[[9]](https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e)

For our example, we won't sweat it for now and stick to 30 neurons.

## Implementation

For our basic neural network, we have a composition of functions:

```python
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))  # same as ReLU; all negative values are 0
    res = res@w2 + b2
    return res

w1 = init_params((28*28, 30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```

Here we have our input layer that outputs 30 features into our hidden layer, which then 
collapses into one output layer. As expected, PyTorch can make our lives easier with classes: 

```python
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```

Combined with PyTorch's `Learner` class and plotting our learning on a graph, we have: 

```python
learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(40, 0.1)
plt.plot(L(learn.recorder.values).itemgot(2))
plt.show()
```

```
epoch     train_loss  valid_loss  batch_accuracy  time
0         0.285860    0.624261    0.500000        00:00
1         0.138273    0.429865    0.816337        00:00
2         0.077861    0.320763    0.919802        00:00
3         0.052002    0.282444    0.945050        00:00
4         0.039859    0.263411    0.959901        00:00
5         0.033560    0.251736    0.968317        00:00
6         0.029884    0.243634    0.971287        00:00
7         0.027463    0.237506    0.972772        00:00
8         0.025704    0.232564    0.974257        00:00
9         0.024335    0.228396    0.976238        00:00
10        0.023224    0.224762    0.977723        00:00
11        0.022296    0.221510    0.979208        00:00
12        0.021507    0.218552    0.979703        00:00
13        0.020824    0.215822    0.980198        00:00
14        0.020227    0.213281    0.980198        00:00
15        0.019698    0.210896    0.981188        00:00
16        0.019225    0.208641    0.982673        00:00
17        0.018798    0.206498    0.982673        00:00
18        0.018410    0.204453    0.983663        00:00
19        0.018056    0.202493    0.983663        00:00
20        0.017730    0.200608    0.984653        00:00
21        0.017428    0.198789    0.984653        00:00
22        0.017148    0.197033    0.985644        00:00
23        0.016886    0.195334    0.986139        00:00
24        0.016641    0.193692    0.986139        00:00
25        0.016410    0.192110    0.986634        00:00
26        0.016192    0.190585    0.986634        00:00
27        0.015986    0.189119    0.986634        00:00
28        0.015791    0.187706    0.986634        00:00
29        0.015606    0.186342    0.987129        00:00
30        0.015430    0.185024    0.987129        00:00
31        0.015262    0.183747    0.987624        00:00
32        0.015102    0.182510    0.988119        00:00
33        0.014948    0.181305    0.988119        00:00
34        0.014801    0.180131    0.988119        00:00
35        0.014660    0.178985    0.988119        00:00
36        0.014525    0.177864    0.988119        00:00
37        0.014394    0.176766    0.988119        00:00
38        0.014269    0.175690    0.988119        00:00
39        0.014147    0.174633    0.988119        00:00
```

![final_learn](images/learning_plot.png)

To view our final accuracy, we write:

```python
print(learn.recorder.values[-1][2])
```

`0.9881188273429871`

### Increasing Depth

While two linear layers is nice, we can go further. Recall that the composition of linear 
functions is just another linear function; so as long as we don't have linear layers that 
are directly adjacent to each other, we're in the clear to add as many layers as we want. 

The question arises: why bother going deeper? Didn't we just prove through the Universal 
Approximation Theorem that one hidden nonlinear layer is sufficient to approximate any 
function?

True. But remember this was only the case for a sufficiently large amount of parameters. 
The Universal Approximation Theorem also has variants - bounded depth and arbitrary width 
(the one we just considered), and bounded width with arbitrary depth. What does that mean? 
By adding more layers, we can reduce the amount of parameters necessary, and improve 
the performance of our model and still be able to approximate any function. 
That is, smaller matrices with many layers is more performant that larger matrices with fewer 
layers in practice. We can do more training with less memory with this structure; indeed, 
it is hard to find neural networks with only one hidden nonlinear layer nowadays.

Now, we'll train an 18-layer model using the same approach in Chapter 1: 

```python
dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet18, pretrained=False, loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```

```
epoch     train_loss  valid_loss  accuracy  time    
0         0.062832    0.014969    0.995584  00:06   
```

Don't sweat `fit_one_cycle()` too much; it basically analyzes the validation/test loss function 
for clues on avoiding underfitting and overfitting, as well as increasing/decreasing the 
learning rate/momentum to optimize training times. 
[[10]](https://sgugger.github.io/the-1cycle-policy.html)

Anyways, the point of this is we got something closs to 100% accuracy with only one epoch!

