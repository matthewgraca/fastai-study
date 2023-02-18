from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import * 
from time import sleep 

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image') 

urls = search_images('bird photos', max_images=1)
print(urls[0])

dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

im = Image.open(dest)
im.to_thumb(256,256).show() # images uses pillow under the hood; use those library's docs for questions

download_url(search_images('plane photos', max_images=1)[0], 'plane.jpg', show_progress=False) 
Image.open('plane.jpg').to_thumb(256,256).show()

# Step 1: download images of birds and not birds
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
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Number of broken images: {len(failed)}")

# DataLoaders is an object containing the training set and validation set
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,  
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label, 
    item_tfms=[Resize(192, method='squish')] 
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# Step 3: test the model, and build your own with different images
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
print(f"Probability it's a plane: {probs[1]:.4f}")
