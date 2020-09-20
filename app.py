# Importing required libraries
import streamlit as st
import tensorflow
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageFile
import tensorflow.keras.applications.mobilenet
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from streamlit import caching

caching.clear_cache()

st.set_option('deprecation.showfileUploaderEncoding', False)

START = "startseq"
STOP = "endseq"
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
max_length = 34
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input


@st.cache(allow_output_mutation=True)
def inceptionV3_model():
    encode_model = InceptionV3(weights='imagenet')
    encode_model = Model(encode_model.input, encode_model.layers[-2].output)
    encode_model._make_predict_function()
    return encode_model


@st.cache(allow_output_mutation=True)
def caption_generator_model():
    model = tf.keras.models.load_model('cap_model50.h5', compile=False)
    model._make_predict_function()
    return model


@st.cache(allow_output_mutation=True)
def wordtoindex():
    wordtoidx = pickle.load(open('WTI.pkl', 'rb'))
    return wordtoidx


@st.cache(allow_output_mutation=True)
def indextoword():
    idxtoword = pickle.load(open('ITW.pkl', 'rb'))
    return idxtoword


def about():
    st.write(
        '''
        **Image Caption Generator** Image captioning is the process of generating textual description of an image.
        It uses both Natural Language Processing and Computer Vision to generate the captions.
        It uses Artificial neural networks such as CNN and RNN.
        The model was trained on Flickr8k image dataset.
        Since the model was only trained on jpg image dataset, the model can take only jpg image as an input.


        The 2 pretrained models used here are:
            1. inceptionV3 model which is trained on imagenet dataset
            2. GloVe model which is an unsupervised learning algorithm. 

Read more :point_right:https://ai.googleblog.com/2016/09/show-and-tell-image-captioning-open.html
        ''')


def main():
    st.title("Image caption generator App :sunglasses: ")
    st.write("**Neural networks used: CNN and RNN:**")

    activities = ["Home", "Know more"]
    choice = st.sidebar.selectbox("About:", activities)

    if choice == "Home":

        st.write("Go to the About section from the sidebar to learn more about the project:")
        image_file = st.file_uploader("Upload a jpg image and then click on the process button below: ", type=['jpg'])
        if image_file is not None:
            image = Image.open(image_file)
            if st.button("Process"):

                encode_model = inceptionV3_model()
                model = caption_generator_model()

                def encodeImage(img):
                    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
                    x = tensorflow.keras.preprocessing.image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    x = encode_model.predict(x)
                    x = np.reshape(x, OUTPUT_DIM)
                    return x

                wordtoidx = wordtoindex()
                idxtoword = indextoword()

                def generateCaption(photo):
                    in_text = START
                    for i in range(max_length):
                        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
                        sequence = pad_sequences([sequence], maxlen=max_length)
                        yhat = model.predict([photo, sequence], verbose=0)
                        yhat = np.argmax(yhat)
                        word = idxtoword[yhat]
                        in_text += ' ' + word
                        if word == STOP:
                            break
                    final = in_text.split()
                    final = final[1:-1]
                    final = ' '.join(final)
                    return final

                st.image(image, use_column_width=True)
                img = encodeImage(image).reshape((1, OUTPUT_DIM))
                prediction = generateCaption(img)
                st.subheader("The generated caption is as follows:")
                st.success(prediction)

    elif choice == "Know more":
        about()


if __name__ == "__main__":
    main()

