from fastai.vision.all import * 

# downloads dataset that comes with the fastai library
path = untar_data(URLs.PETS)/'images'   

# our data has cat pics start w/ upper case and dog pics w/ lowercase,
# where the name of the file is the label
# e.g. Bengal_100.jpg := cat, chihuahua_14.jpg := dog
'''
This is why I hate python, you literally can't infer anything by reading code.
Like what does is_cat return? the upper of the first index of an array of 
what? If it just says x is a char array it would be easy to tell but no... 
I have to read a fking book that explains the code to know what it does...
A function whose domain and codomain are not defined IS NOT A REAL FUNCTION, 
IT'S A BLACK BOX WHOSE INPUTS AND OUTPUTS ARE BLACK BOXES.
'''
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

# uses resnet34, a pretrained model
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

# test pic of cat
img = PILImage.create('test_images/cat.jpeg')
is_cat,_,probs = learn.predict(img)
print(f"Is this cat a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

# test pic of mouse
img = PILImage.create('test_images/mouse.jpg')
is_cat,_,probs = learn.predict(img)
print(f"Is this mouse a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

# test pic of toyota tacoma
img = PILImage.create('test_images/toyota_tacoma.jpg')
is_cat,_,probs = learn.predict(img)
print(f"Is this toyota tacoma a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

# test pic of hourglass
img = PILImage.create('test_images/hourglass.jpg')
is_cat,_,probs = learn.predict(img)
print(f"Is this hourglass a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

# test pic of a dog
img = PILImage.create('test_images/dog.jpeg')
is_cat,_,probs = learn.predict(img)
print(f"Is this dog a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

'''
Explanation of performace:
The resnet34 model is trained with the Oxford-IIIT Pet Dataset, which contains 
images of just dogs and cats. The model can only think of images in terms of 
dogs and cats, so when it is presented with a picture that is decidedly 
neither one of those two, it's basically just guessing. That is why it is so 
confident with pictures of cats and dogs, but has no understanding of anything 
else since it can only respond with dog or cat. Imagine if you're forced to 
answer if a toyota tacoma is either a cat or dog?
'''
