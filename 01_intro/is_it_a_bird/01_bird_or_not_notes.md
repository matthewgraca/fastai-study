# Notes for helloworldfastai.py

Adapted from [this tutorial](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data).

## Code Walkthrough

### Imports

This library allows us to use duckduckgo search in our code

`from duckduckgo_search import ddg_images`

This library contains the core of fastai, like their super special list class `L` and `itemgot`

`from fastcore.all import *`

This library allows us to download the pictures we searched using duckduckgo

`from fastdownload import download_url`

This library is responsible for the actual deep learning model

`from fastai.vision.all import *`

This library ensures that server overloading doesn't occur from pinging it a bunch

`from time import sleep`

### Functions and Testing

First we define a function that gets a query, then fills a list of URLs from ddg of that query.

```
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image' 
```

Here we test this out by searching for bird photos, getting a URL of that query, then printing it out. Then we download the image associated with that URL, and open it to confirm. The same is done for a plane photo.

```
urls = search_images('bird photos', max_images=1)
print(urls[0])

dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

im = Image.open(dest)
im.to_thumb(256,256).show() 

download_url(search_images('plane photos', max_images=1)[0], 'plane.jpg', show_progress=False) 
Image.open('plane.jpg').to_thumb(256,256).show()
```

### Downloading Images of Birds and Not Birds

First we create folders for plane and bird photos. We want the subject of these images in different environments, so we append our searches with things like sun, shade, etc.

By default `search_images` gets 30 URLs, so we'll have a total of 180 images. We also resize the images so they're small and uniform.

```
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
```

### Fit our Data to the Model

Before we use a model, we need to format our inputs so the model knows what to do with them.

First, we prune images that did not download correctly. We use `get_image_files()` to grab the image files in a folder, then use `fastat.vision.utils` containing `verify_images()`, which returns a list of images that failed to open. Finally, we use fastai's `L.map()` to unlik all the images we identified that failed to open.

```
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Number of broken images: {len(failed)}")
```

Next, we use `DataBlock`, which is a `DataLoaders` object that defines the data being used for the model.
- `blocks` defines the inputs and outputs; in this case, `ImageBlock` refers to the inputs as images, and `CategoryBlock` refers to the outputs as categories.
- `get_items` gets the inputs to our model, which are image files; thus `get_image_files`
- `splitter` determines how we should divide the validation and test set. Here we use a defined random seed, and randomly divide our data into 80% test and 20% validation.
- `get_y` is our labels (y values), which are the names of the parent of each file. This is because we put all our bird files into a folder that demarcates them as birds; so this is how we know those files are of birds.
- `item_tfms` transforms the inputs; here we "squish" them (as opposed to cropping them) into 192x192 pixels.

```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42), 
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')] 
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```

Finally, we define the model we are using. We use `resnet18`, a computer vision model that is pretrained with its own data and parameters. As such, we need to use fastai's `fine_tune()` function, that does weight-adjusting so that this model can recognize our particular dataset.

`vision_learner` is just fastai's Learner object; gets our formatted data, pretrained model, and metric for identifying the success/failure of the architecture used.

```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

### Testing the Model

Now is the time to actually test the model. Note that you will have to download all the test images yourself, it will not be on this repo.

We use `Learner.predict()`, which returns a triple: class, label, and probabilities for a given item. In this case, the probability that an image is a bird. Note that `probs` is sorted alphabetically, so since we have data for birds and planes, `probs[0]` will be the birds prediction and `probs[1]` will be planes prediction.

```
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
print(f"Probability it's a plane: {probs[1]:.4f}")
```
