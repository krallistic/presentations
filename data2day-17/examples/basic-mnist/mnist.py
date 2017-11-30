import tensorflow as tf
import argparse
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data

# reset everything to rerun in jupyter


# config
batch_size = 100
learning_rate = 0.01
training_epochs = 10



def main(_):
    # load mnist data set
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # input images
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

    # model parameters will change during training so we use tf.Variable
    W = tf.Variable(tf.zeros([784, 10]))

    # bias
    b = tf.Variable(tf.zeros([10]))

    # implement model
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # specify cost function
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # specify optimizer
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
    
    start_training_time = time.time()
    with tf.Session() as sess:
    # variables need to be initialized before we can use them
        sess.run(tf.initialize_all_variables())

        # perform training cycles
        for epoch in range(training_epochs):
                
            # number of batches in one epoch
            batch_count = int(mnist.train.num_examples/batch_size)
                
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                    
                # perform the operations we defined earlier on batch
                sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
                    
            if epoch % 2 == 0: 
                print "Epoch: ", epoch 
        print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    print("Training Time: %3.2fs" % float(time.time() - start_training_time))
    print("Done Training example")



if __name__ == '__main__':
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
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    