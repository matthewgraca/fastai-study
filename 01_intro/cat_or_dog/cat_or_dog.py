from fastai.vision.all import * 

path = untar_data(URLs.PETS)/'images'   

'''
This is why I hate python, you literally can't infer anything by reading code.
Like what does is_cat return? the upper of the first index of an array of 
what? If it just says x is a char array it would be easy to tell but no... 
I have to read an f-ing book that explains the code to know what it does...
A function whose domain and codomain are not defined IS NOT A REAL FUNCTION, 
IT'S A BLACK BOX WHOSE INPUTS AND OUTPUTS ARE BLACK BOXES.
'''
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

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
