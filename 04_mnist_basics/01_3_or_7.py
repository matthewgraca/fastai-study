from fastai.vision.all import *
from fastbook import * 
import matplotlib.pyplot as plt
import webbrowser

# download a sample of mnist data (3s and 7s)
path = untar_data(URLs.MNIST_SAMPLE)
'''
print(path.ls())
print((path/'train').ls())
'''

# saves and sorts the image names, shows top 10 of the 3s
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
'''
print(threes[:10])
'''

# visualization of an image as data
'''
# grab and show a data point for 3
im3_path = threes[1]
im3 = Image.open(im3_path)
im3.show()

# as numpy array
print(array(im3)[4:10,4:10]) # views rows from [4,10) and cols from [4,10)
# as pytorch tensor
print(tensor(im3)[4:10,4:10])
# use pandas dataframe to color code values using a gradient; shows how an image is conjured from pixel values
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:26,4:22])
df_formatted = df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
# open image generated by the dataframe in browser
with open('pixel_image.html', 'w') as image:
    df_formatted.to_html(image)
filename = 'pixel_image.html'
webbrowser.open_new_tab(filename)
'''

# first try - pixel similarity
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
'''
print(len(three_tensors),len(seven_tensors))
show_image(three_tensors[1], cmap='binary')
plt.show()
'''

# compute average of each pixel by stacking the tensors into a rank-3 tensor (3D tensor)
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

# investigating tensor jargon
'''
print(stacked_threes.shape) # 6131 28x28 images stacked as a 3D tensor
print(len(stacked_threes.shape))    # length of a tensor's shape = rank of the tensor ->
print(stacked_threes.ndim)          # rank of a tensor (ndim) = number of axes in a tensor 
'''

# compute the "ideal" 3 and 7
mean3 = stacked_threes.mean(0) 
mean7 = stacked_sevens.mean(0)
'''
show_image(mean3, cmap='binary')
plt.show()

show_image(mean7, cmap='binary')
plt.show()
'''

# sample 3 and 7
a_3 = stacked_threes[1]
a_7 = stacked_sevens[1]

# distance b/t a datapoint and the ideals
dist3_abs = (a_3 - mean3).abs().mean()
dist3_sqr = ((a_3 - mean3)**2).mean().sqrt()

dist7_abs = (a_3 - mean7).abs().mean()
dist7_sqr = ((a_3 - mean7)**2).mean().sqrt()

'''
print(dist3_abs, dist3_sqr)
print(dist7_abs, dist7_sqr)
print(F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt())
'''

# test pixel similarity method on validation set
valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255

valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_3_tens.float()/255
'''
print(valid_3_tens.shape, valid_7_tens.shape)
'''

def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
'''
print(mnist_distance(a_3, mean3))
'''

'''
valid_3_dist = mnist_distance(valid_3_tens, mean3)
print(valid_3_dist, valid_3_dist.shape)
'''

def is_3(x): return mnist_distance(x, mean3) < mnist_distance(x, mean7)

'''
print(is_3(a_3), is_3(a_3).float())
print(is_3(valid_3_tens))
'''

# calculate metrics
accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()
'''
print(accuracy_3s, accuracy_7s, (accuracy_3s+accuracy_7s)/2)
'''

# mnist loss function
# vectorize data and prepare labels for training set
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
'''
print(tensor([1]*len(threes) + [0]*len(sevens)))
print(train_y)
print(train_x.shape, train_y.shape)
'''

# create a list of pairs; (image, label)
dset = list(zip(train_x, train_y))
x,y = dset[0]
'''
print(x.shape, y)
'''

# vectorize and prepare labels for validation set
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

# begin sgd

# init random parameters
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
weights = init_params((28*28, 1))
bias = init_params(1)

# predict (for one image)
'''
print((train_x[0]*weights.T).sum() + bias)
'''

# predictions for all of the training data
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
'''
print(preds)
'''

# check accuracy of predictions
corrects = (preds>0.0).float() == train_y
'''
print(corrects)
print(corrects.float().mean().item())
'''

# see effect of changing one weight
with torch.no_grad(): weights[0] *= 1.0001
preds = linear1(train_x)
'''
print(((preds>0.0).float() == train_y).float().mean().item())
'''

# first attempt loss function definition
trgts = tensor([1,0,1])
prds = tensor([0.9,0.4,0.2])

'''
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

print(torch.where(trgts==1, 1-prds, prds))
print(mnist_loss(prds,trgts))
print(mnist_loss(tensor([0.9,0.4,0.8]),trgts))
'''

# sigmoid
def sigmoid(x): return 1/(1+torch.exp(-x))
'''
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
plt.show()
'''

# second attempt at loss function definition
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

# mini-batch demo
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
'''
print(list(dl))
'''
ds = L(enumerate(string.ascii_lowercase))
dl = DataLoader(ds, batch_size=6, shuffle=True)
'''
print(ds)
print(list(dl))
'''

###
# full implementation of gradient descent

# init params
weights = init_params((28*28,1))
bias = init_params(1)

# format training dataset into batches of 256
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

xb,yb = first(dl) # fastcore.basics function. getter for first element
'''
print(xb.shape, yb.shape)
'''

# test with mini-batch
batch = train_x[:4]
'''
print(batch.shape)
'''

preds = linear1(batch)
'''
print(preds)
'''

loss = mnist_loss(preds, train_y[:4])
'''
print(loss)
'''

loss.backward()
'''
print(weights.grad.shape, weights.grad.mean(), bias.grad)
'''

# put into a function
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

# test calc_grad()
calc_grad(xb, yb, linear1)
'''
print(weights.grad.mean(), bias.grad)
'''

calc_grad(xb, yb, linear1)
'''
print(weights.grad.mean(), bias.grad)
'''

weights.grad.zero_()
bias.grad.zero_()

# epoch function definition
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()

# accuracy of mini-batch test
'''
print((preds>0.0).float() == train_y[:4])
'''

# accuracy function definition
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

'''
print(batch_accuracy(linear1(batch), train_y[:4]))
'''

# validation epoch function definition
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

'''
print(validate_epoch(linear1))
'''

# run epochs
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
'''
print(validate_epoch(linear1))
'''

'''
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
print()
'''

# sgd with pytorch modules
linear_model = nn.Linear(28*28, 1)
w,b = linear_model.parameters()
'''
print(w.shape, b.shape)
'''

# creating optimizer
class BasicOptim:
    def __init__(self, params, lr): self.params,self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None

opt = BasicOptim(linear_model.parameters(), lr)

def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

'''
print(validate_epoch(linear_model))
'''

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
    print()

'''
train_model(linear_model, 20)
'''

# using fastai libraries for sgd
linear_model = nn.Linear(28*28, 1)
opt = SGD(linear_model.parameters(), lr)
'''
train_model(linear_model, 20)
'''

# using fastai to further compress code to train
dls = DataLoaders(dl, valid_dl)
learn = Learner(dls, nn.Linear(28*28, 1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
'''
learn.fit(10, lr=lr)
'''

# adding nonlinear layers to neural net
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))  # same as ReLU; all negative values are 0
    res = res@w2 + b2
    return res

w1 = init_params((28*28, 1))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)

'''
plot_function(F.relu)
plt.show()
'''

# doing the same with PyTorch instead
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

'''
learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(40, 0.1)
plt.plot(L(learn.recorder.values).itemgot(2))
plt.show()
print(learn.recorder.values[-1][2])
'''

# example with many layers
dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet18, pretrained=False, loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
