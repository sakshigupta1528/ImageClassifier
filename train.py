# TODO: Make all necessary imports.
import warnings
warnings.filterwarnings('ignore')


import time
import numpy as np
import matplotlib.pyplot as plt
import json

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data_dir', type=str)
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='train_checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    print('Using:')
    print('\t\u2022 TensorFlow version:', tf.__version__)
    print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
    dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)
    # TODO: Create a training set, a validation set and a test set.
    training_set, validation_set, testing_set = dataset['train'],dataset['validation'], dataset['test']
    dataset_info
    # TODO: Get the number of examples in each set from the dataset info.
    num_training_examples = dataset_info.splits['train'].num_examples
    num_validation_examples = dataset_info.splits['validation'].num_examples
    num_test_examples = dataset_info.splits['test'].num_examples


    # TODO: Get the number of classes in the dataset from the dataset info.
    num_classes = dataset_info.features['label'].num_classes

    print('There are {:,} classes in our dataset'.format(num_classes))

    print('\nThere are {:,} images in the test set'.format(num_test_examples))
    print('There are {:,} images in the validation set'.format(num_validation_examples))
    print('There are {:,} images in the training set'.format(num_training_examples))
    shape_images = dataset_info.features['image'].shape

    for image, label in training_set.take(3):
        image = image.numpy().squeeze()
        label = label.numpy()
        print('The label of this image is:', label)

    print('\nThe images in our dataset have shape:', shape_images)
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
    feature_extractor.trainable = False
    layer_neurons = [650, 330, 250]

    dropout_rate = 0.2

    model = tf.keras.Sequential()

    model.add(feature_extractor)

    #for neurons in layer_neurons:
    #    model.add(tf.keras.layers.Dense(neurons, activation = 'relu'))
    #    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(102, activation = 'softmax'))

    model.summary()
    # TODO: Plot the loss and accuracy values achieved during training for the training and validation set.

    # model.compile(optimmizer = 'adam',
    #               loss = 'sparse_categorical_crossentropy',
    #               metrics = ['accuracy'])




    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', 'categorical_accuracy']
    )


    with tf.device('/GPU:0'):
        EPOCHS = 5

        history = model.fit(training_batches,
                            epochs=EPOCHS,
                            validation_data=validation_batches)

    #EPOCHS = 5

    #history = model.fit(training_batches,
    #                       epochs=EPOCHS,
    #                       validation_data=validation_batches)
    loss, accuracy, categorical_accuracy = model.evaluate(testing_batches)

    print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
    print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))
    print('Categorical Accuracy on the TEST Set: {:.3%}'.format(categorical_accuracy))

batch_size = 32
image_size = 224


image_gen_train = ImageDataGenerator(rescale = 1./255,
                                     rotation_range = 45,
                                     width_shift_range = 0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')



def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

training_batches = training_set.cache().shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = testing_set.map(format_image).batch(batch_size).prefetch(1)


if __name__ == "__main__":
    main()

