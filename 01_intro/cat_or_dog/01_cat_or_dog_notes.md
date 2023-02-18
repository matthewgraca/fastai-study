# Notes for cat_or_dog.py

## Code Walkthrough

First, we download a dataset that comes with the fastai library

`path = untar_data(URLs.PETS)/'images'` 

Next, we check each file that we downloaded. The dataset contains two kinds of data: pictures of dogs, and pictures of cats. The way that the authors differentiated the two was through having every file with a cat picture have its name begin with an uppercase letter, and every file with a dog picture have its name begin with a lowercase letter.

For example, `Bengal_100.jpg` would be a cat picture, while `chihuahua_14.jpg` would be a dog picture.

`def is_cat(x): return x[0].isupper()`

Here, we use `ImageDataLoaders`, a fastai class for handling data for image classification models, to initialize some things about our model.
- We define a `path` to the images, and a function that opens those images.
- The percent of the data being used as the validation set.
- Set seed to (deterministically) randomly pick the pictures that will be used for that validation set. 
- We pass in the function for determining which images we'll train the model with from the set of all pictures.
- Then resizing the pictures to something small and uniform.

```
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
```

Now that we've formatted and defined the data the model will be using, we need to define the actual model itself.
- We pass in `dls` so the model knows what data to use and how.
- We use the `resnet34` pretrained model, which is pretty good and has done a lot of the work for us already.
- Tell the model to use `error_rate` as the metric for counting a properly classified image.

Finally, we fine tune the model; using a pretrained model, it comes with training that we don't need on objects we don't care about; fine tuning ensures that the model fits our particular use-case.

```
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

## Performance Notes

Note that the test images and dataset are not included because then the size of this repo would explode :P

If you run this code (with all the test images downloaded as well), you'll notice that the model doesn't seem to perform well when an image of anything other than a dog or a cat is being tested.

The resnet34 model is trained with the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains images of just dogs and cats. The model can only think of images in terms of dogs and cats, so when it is presented with a picture that is decidedly neither one of those two, it's basically just guessing. That is why it is so confident with pictures of cats and dogs, but has no understanding of anything else since it can only respond with dog or cat. It has never seen a toyota tacoma, yet it's being asked if that thing is a dog or a cat (neither is not an answer).

