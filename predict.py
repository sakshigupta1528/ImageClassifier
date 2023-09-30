import numpy as np
import json
from collections import OrderedDict
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='test_images/hard-leaved_pocket_orchid.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    return parser.parse_args()


def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255 
    image = image.numpy().squeeze()
    return image

from PIL import Image
image_size = 224
image_path = 'test_images/hard-leaved_pocket_orchid.jpg'
def predict(image, model, top_k): 
    print(type(image))
    image = Image.open(image_path)
    print(type(image))
    image = np.asarray(image)
    print(type(image))
    image = tf.cast(image, tf.float32)
    print(type(image))
    image = tf.image.resize(image, (image_size, image_size))
    print(type(image))
    image /= 255 #Tensor: rescaling images to be between 0 and 1
    print(type(image))
    image = image.numpy().squeeze()
    print(type(image), image.shape)
    
    image = np.expand_dims(image, axis = 0)
    print(type(image), image.shape)
    
    
    ps = model.predict(image)[0] #ps is a list of lists, we have only one, we lelect that one
    #print(model.predict(image).shape)
    #print(model.predict(image))
    #print(ps)
    #print(np.argsort(ps))
    probabilities = np.sort(ps)[-top_k:len(ps)]
    prbabilities = probabilities.tolist()
    print(prbabilities)
    classes = np.argpartition(ps, -top_k)[-top_k:] 
    classes = classes.tolist()
    names = [class_names.get(str(i + 1)).capitalize() for i in (classes)]
    print(names)
    #classes = idx[np.argsort(ps[idx])][::-1]  #Name Class Indices sorted by value from largest to smallest
    #print(idx)
    #class_names = [class_names.get(str(i)).capitalize() for i in classes]
    ps_cl = list(zip(prbabilities, names))
    print(ps_cl)
    return probabilities, names
