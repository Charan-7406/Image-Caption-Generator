# Image-Caption-generator
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions. It uses Artificial neural networks such as CNN and RNN.The model was trained on Flickr8k image dataset.Since the model was only trained on jpg image dataset, the model can take only jpg image as an input.
The 2 pretrained models used here are:
            1. inceptionV3 model which is trained on imagenet dataset
            2. GloVe model which is an unsupervised learning algorithm. 
            
 
 # System requirements
 A good CPU and a GPU with atleast 8GB memory.
 Atleast 8GB of RAM.
 Active internet connection so that keras can download inceptionv3/vgg16 model weights and also glove model.
 If the system gpu not good enough use google colab.
 
# Libraries required
* Keras 
* Tensorflow 
* tqdm
* numpy
* pickle
* PIL
* glob
* tqdm
* streamlit (used for the deployment of the model)
 
 # Dataset:
 * Flicker8k dataset: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
 * Flicker8k text : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
 
 # Downloading glove embedding  file on google colaboratory.
 * Download the zip file: !wget http://nlp.stanford.edu/data/glove.6B.zip
 * Unzip it: !unzip glove*.zip
 
 # About the model created:
 The loss value of  around **2.1** has been achieved , where the model was trained for 50 epochs  gives decent results. You can check out some examples of the caption generated on the test model in the snaps_generated_caption_for_testimage folder.The rest of the examples are in the jupyter notebook. You can run the Jupyter Notebook and try out your own examples.In the notebook i have trained model for 20 epochs since my gpu isnt good enough. I have uploaded model trained on 50 epochs using google colab.Everything is implemented in the Jupyter notebook which will hopefully make it easier to understand the code.
  *unique.p* is a pickle file which contains all the unique words in the vocabulary. 


 
 
 

