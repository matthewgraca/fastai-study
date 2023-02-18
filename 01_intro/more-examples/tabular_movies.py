# recommends a movie to watch based off other movies a person has already seen

# uses collaborative filtering, a process in which a hole in a user's data is filled
# using the data from similar users.
# e.g. person A and B like "1", person A likes "2" but person B has no data. It is
# decided that person B is like person A, so the model assumes person B also likes "2"
from fastai.collab import *
from fastai.data.external import * 

path = untar_data(URLs.ML_SAMPLE)

dls = CollabDataLoaders.from_csv(path/'ratings.csv')

learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10)
learn.show_results()
# loss can be larger than 1 since we're working with continuous data; not just yes/no;
# now we have to measure how wrong we are by how "far" we are from the actual answer;
# even with a good model if we're off a little bit everywhere, it adds up.
