import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical


# ## Preparing the data


reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)


# ### Counting word frequency


from collections import Counter

total_counts = Counter()        # bag of words

for idx, row in reviews.iterrows():
    total_counts.update(row[0].split(" "))         

# Let's keep the first 10000 most frequent words. As Andrew noted, most of the words in the vocabulary are rarely used so they will have little effect on our predictions. Below, we'll sort `vocab` by the count value and keep the 10000 most frequent words.


vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]

# What's the last word in our vocabulary? We can use this to judge if 10000 is too few. If the last word is pretty common, we probably need to keep more words.

# Now for each review in the data, we'll make a word vector. First we need to make a mapping of word to index, pretty easy to do with a dictionary comprehension.

word2idx = {}             ## create the word-to-index dictionary here
for i, words in enumerate(vocab):
    word2idx[words] = i


# ### Text to vector function
# 
# Now we can write a function that converts a some text to a word vector. The function will take a string of words as input and return a vector with the words counted up. Here's the general algorithm to do this:


def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return word_vector

# Now, run through our entire review data set and convert each review to a word vector.

word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

# ### Train, Validation, Test sets
# 
# Now that we have the word_vectors, we're ready to split our data into train, validation, and test sets. Remember that we train on the train data, use the validation data to set the hyperparameters, and at the very end measure the network performance on the test data. Here we're using the function `to_categorical` from TFLearn to reshape the target data so that we'll have two output units and can classify with a softmax activation function. We actually won't be creating the validation set here, TFLearn will do that for us later.

Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9
train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)

# Network building
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()

    net = tflearn.input_data([None,len(vocab)])
    net = tflearn.fully_connected(net, 5, activation='ReLU')
    net = tflearn.fully_connected(net, 5)
    net = tflearn.fully_connected(net, 2, activation='softmax')    
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model

# ## Intializing the model
# 
# Next we need to call the `build_model()` function to actually build the model. 
# > **Note:** You might get a bunch of warnings here. TFLearn uses a lot of deprecated code in TensorFlow. Hopefully it gets updated to the new TensorFlow version soon.

model = build_model()


# ## Training the network
# 
# Now that we've constructed the network, saved as the variable `model`, we can fit it to the data. Here we use the `model.fit` method. You pass in the training features `trainX` and the training targets `trainY`. Below I set `validation_set=0.1` which reserves 10% of the data set as the validation set. You can also set the batch size and number of epochs with the `batch_size` and `n_epoch` keywords, respectively. Below is the code to fit our the network to our word vectors.

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=10)


# ## Testing

predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)

def magic_predict(text):
    positive_prob = model.predict([text_to_vector(text.lower())])[0][1]
    print('P(Positive) = {:.3f} :'.format(positive_prob), 
      'Positive' if positive_prob > 0.5 else 'Negative')
