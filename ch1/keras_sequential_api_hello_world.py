# hello world Keras sequential API model for MNIST
# adapted from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb

#import required libraries

import tensorflow as tf
import pydotplus
from tensorflow.keras.utils import plot_model

mnist = tf.keras.datasets.mnist

# define inputs for the model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# define layers for the hello world model

hello_world_model = tf.keras.models.Sequential([ 
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(128, activation='relu'), 
  tf.keras.layers.Dropout(0.15), 
  tf.keras.layers.Dense(10) 
])

# compile the hello world model, including specifying the loss function, optimizer, and metrics

hello_world_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 

# train model

history = hello_world_model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=10,
                    validation_split=0.15)

# assess performance of the model
                    
test_scores = hello_world_model.evaluate(x_test,  y_test, verbose=2) 
print('Loss for test dataset:', test_scores[0])
print('Accuracy for test dataset:', test_scores[1])
