import os #attempt to avoid error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,)

n_train = mnist.train.num_examples # 55,000
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000

#Three hidden layers
n_hid1 = 1000
n_hid2 = 250
n_hid3 = 100 

n_iterations = 1000
n_batch_size = 100
dropout = 0.6

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float")

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hid1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hid2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hid3])),
    'out': tf.Variable(tf.constant(0.1, shape=[10]))
}

weights = {
    'w1': tf.Variable(tf.truncated_normal([784, n_hid1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hid1, n_hid2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hid2, n_hid3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hid3, 10], stddev=0.1)),
}


layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(n_batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})

    # print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
print("\nAccuracy on test set:", test_accuracy)

img = np.invert(Image.open("test_img.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))

