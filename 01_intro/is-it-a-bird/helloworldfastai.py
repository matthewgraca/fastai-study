from duckduckgo_search import ddg_images    # search images via duckduckgo.com
from fastcore.all import *  # core libraries, like L (their super special list class) and itemgot
from fastdownload import download_url   # downloading goods
from fastai.vision.all import * # viewing images
from time import sleep  # prevent server overload from pinging it a bunch

# this function gets a query and fills a list of images from ddg of that query
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image') # a list containing urls of ddg image results

# this part shows off image downloading and opening, and ultimately what we'll use to see if our model works
urls = search_images('bird photos', max_images=1)
print(urls[0])

dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

im = Image.open(dest)
im.to_thumb(256,256).show() # images uses pillow under the hood; use those library's docs for questions

download_url(search_images('plane photos', max_images=1)[0], 'plane.jpg', show_progress=False)    # i have no clue what search_images()[0] does
Image.open('plane.jpg').to_thumb(256,256).show()

# Step 1: download images of birds and not birds

# create folders for plane and bird photos of different kinds
# total of 180 images
searches = 'plane','bird'
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# Step 2: train model to determine the difference

# prune images that did not download correctly
# uses fastai.vision.utils 
# verify_images() returns a list of images that failed to open
# get_image_files grabs image files in a folder
# L.map() to unlinks all the images that failed to open
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Number of broken images: {len(failed)}")

# DataLoaders is an object containing the training set and validation set
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # inputs = images, outputs = categories
    get_items=get_image_files,  # get the inputs to our model (image files)
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # randomly split our data into 20% validation and 80% training
    get_y=parent_label, # the labels (y values) are the names of the parent of each file
    item_tfms=[Resize(192, method='squish')] # resize all inputs by squishing them  (as opposed to say, cropping) into 192x192 pixels
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

# model - uses resnet18, a cv model
# model is fine tuned with fastai's fine_tune(), which automatically uses best practices for fine tuning a pre-trained model
# basically, it's weight-adjusting since this was trained on a different dataset, and we want this model to recognize our particular dataset
# vision_learner is from fastai's Learner object 
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# Step 3: test the model, and build your own with different images
# Learner.predict() returns a triple: class, label, and probabilities for a given item; in this case, the probability that an image is a bird
# note: probs is sorted alphabetically
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
print(f"Probability it's a plane: {probs[1]:.4f}")
