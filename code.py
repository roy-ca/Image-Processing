from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

import time
import data_helpers

## Creating Timer
beginTime = time.time()


## Parameter definitions
batch_size = 100
learning_rate = 0.005
max_steps = 1000

## data_helpers used because loading and training data is not part for our key goal

data_sets = data_helpers.load_data()

## Define Input Placeholders

image_placeholder = tf.placeholder(tf.float32, shape=[None,3072])
label_placeholder = tf.placeholder(tf.int64, shape=[None])

## Defning variables to optimize
## Two lines of code shown above is that there is a 3072 x 10 matrix of weight parameters, which are all set to 0 in the beginning. 
## Second line defines an array having 10 biases which is to be multiplied with weight

weights = tf.Variable(tf.zeroes([3072, 10]))
biases = tf.Variable(tf.zeroes([10]))

## Computing the product to get the desired label values

logits = tf.matmul(images_placeholder, weights) = biases 

## Calculating loss by using softmax and cross entropy fnctions

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels_placeholder))

## Define training operation

train_step = tf.train.GradientDescentOPtimizer(learning_rate).minimize(loss)

## Comparing models acccuracy

correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)
acccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Running tensor flow graph

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for i in range(max_steps):

		## Creating batches

		indices = np.random.choice(daata_sets['images_train'].shape[0], batch_size)
		images_batch = data_sets['images_train'][indices]
		labels_batch = data_sets['labels_train'][indices]

		if i % 100 == 0:
				train_accuracy = sess.run(accuracy, feed_dict={image_placeholder: images_batch,labels_placeholder:labels_batch})
				print('Step {:5d): training accuracy {:g}'.format(i, train_accuracy))
		sess.run(tain_step, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})

								
test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: data_sets['images_test'], labels_placeholder: data_sets['labels_test']})
print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))




