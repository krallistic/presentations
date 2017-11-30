import tensorflow as tf
import argparse
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data

# reset everything to rerun in jupyter


# config
batch_size = 50
learning_rate = 0.01
training_epochs = 10

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # load mnist data set
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # input images
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

   # first convolutional layer

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    # readout layer

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=learning_rate,
        # optimizer=tf.train.AdamOptimizer
        optimizer="Adam")
    
    prediction = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    start_training_time = time.time()
    with tf.Session() as sess:
    # variables need to be initialized before we can use them
        sess.run(tf.global_variables_initializer())
        # number of batches in one epoch

        batch_count = int(mnist.train.num_examples/batch_size)
        print("BatchCount: "  + str(batch_count))
        # perform training cycles
        for epoch in range(training_epochs):
                

                
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                    
                # perform the operations we defined earlier on batch
                sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
                    
            if epoch % 2 == 0: 
                print("Epoch: ", epoch )
                print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    print("Training Time: %3.2fs" % float(time.time() - start_training_time))
    print("Done Training example")



if __name__ == '__main__':
    print("Starting...")
    tf.reset_default_graph()

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
     
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Amount of epochs"
    )
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("Running with FLAGS: ", FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    