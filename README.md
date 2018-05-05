# Sentiment Analysis:
<br>

![be negative or positive](https://github.com/samiarja/sentiment-analysis/blob/master/neg-pos.PNG)

# Overview:
<br>

***Sentiment analysis also known as opinion mining.***
<br> 

it's a way of determining how **positive** or **negative** the content of a text document is, based on the relative numbers of words it contains that are classified as either positive or negative.
<br>

This technique is refered to the use of natural language processing, text analysis, computational linguistics and it is heavily used in multiple problems such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.

<br>

Here in this example, we want to develop a system that can predict the sentiment of a textual movie review as either **positive** or
**negative**.

## Source:
<br>

Refer to this **arxiv** research paper: [Sentiment Analysis](http://xxx.lanl.gov/abs/cs/0409058)
<br>

# A bit about this model:
<br>

## Libraries used:
<br>

**Open cmd and type:**
<br>
> pip install -r requirements.txt
<br>

It will install all the dependencies for you.
<br>

**or manually the procedure will look like this:**
<br>

Assuming you are using anaconda: currently this is where I do everything, Linux is on to do list.

* [Numpy](https://anaconda.org/anaconda/numpy) : conda install -c anaconda numpy 
* [NLTK](https://anaconda.org/anaconda/nltk) : conda install -c anaconda nltk
* [Keras](https://anaconda.org/conda-forge/keras) : conda install -c conda-forge keras

<br>

# Neural Network architechture:
<br>

![NN-architecture](https://github.com/samiarja/sentiment-analysis/blob/master/multichannel.png)
<br>
Visualization done through keras visualization packages. Simply by typing these two command before return your model.

> **from keras.utils.vis_utils import plot_model**
<br>

> **plot_model(model, to_file='multichannel.png', show_layer_names=True, show_shape=True)**
<br>

# Tokenization procedure:
<br>

The data is already clean for use but we should turn these document to a real token, after that:
<br>

* Split into tokens by white space
* Remove punctuation from each token
* Remove punctuation from each token
* filter out stop words
* filter out short tokens

# Data processing procedure:
<br>

* load all docs in a directory
* walk through all files in the folder
* Skip any reviews in the test set
* Create the full path of the file to open
* Load the doc
* Clean doc
* Add to list
<br>

# Neural Network Consist of a Sequential model which is a linear stack of layers:
<br>

***Three channels***
<br>

* Channel 1: Input -> Embedding -> Conv1D -> Dropout -> MaxPooling1D -> Flatten
* Channel 2: Input -> Embedding -> Conv1D -> Dropout -> MaxPooling1D -> Flatten
* Channel 3: Input -> Embedding -> Conv1D -> Dropout -> MaxPooling1D -> Flatten
* Merge: Concatinate all the Flatten layers
* Apply Dense layer with Relu activation function
* Apply Dense layer with Sigmoid activation function
* Specifying the learning process of:
  * **binary_crossentropy** for the loss(object the model will try to minimize)
  * AdamOptimizer to optimize the model
  * A lists of metrics as Accuracy 
<br>

# After 10 epoch with batch size of 16 the model accuracy was 86%

# Run the model with:
<br>

> **python NN_for_SAnalysis.py** 
<br>

All the credit goes to [machinelearningmastry](https://machinelearningmastery.com)
<br>

First image from [Google](https://www.google.com.au/)
