# predicts if a person is a high-income earner or not

from fastai.tabular.all import *

path = untar_data(URLs.ADULT_SAMPLE)

# y_names isolates the result column (what we want to predict)
dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    # denotes columns that are categorical (contains values with discrete choices)
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    # denotes columns that are continuous (values that are real numbers)
    cont_names = ['age', 'fnlwgt', 'education-num'],
    # data pre-processing: categorify maps the category columns from integer to unique categories
    # fillmissing fills missing continuous columns with the median of that column
    # normalize normalizes continuous variables (subtract the mean, div by the std)
    procs = [Categorify, FillMissing, Normalize])

# there are usually no pre-trained models for tabular data
# instead, use fit_one_cycle to train from scratch (instead of transfer learning)
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(3)
learn.show_results()
