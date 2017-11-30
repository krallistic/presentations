from __future__ import print_function

import tensorflow as tf
import sys
import time

# cluster specification
parameter_servers = ["pc-01:2222"]
workers = [ "pc-02:2222", "pc-03:2222", "pc-04:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)
# config
batch_size = 100
learning_rate = 0.001
training_epochs = 20
logs_path = "/mnist/1"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":
  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):

    global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False)

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

    W1 = tf.Variable(tf.random_normal([784, 100]))
    W2 = tf.Variable(tf.random_normal([100, 10]))
    b1 = tf.Variable(tf.zeros([100]))
    b2 = tf.Variable(tf.zeros([10]))

    #Softmax
    z2 = tf.add(tf.matmul(x,W1),b1)
    a2 = tf.nn.sigmoid(z2)
    z3 = tf.add(tf.matmul(a2,W2),b2)
    y  = tf.nn.softmax(z3)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    grad_op = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = grad_op.minimize(cross_entropy, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.scalar_summary("cost", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)
    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

  with sv.prepare_or_wait_for_session(server.target) as sess:

    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
      batch_count = int(mnist.train.num_examples/batch_size)

      for i in range(batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        _, cost, summary, step = sess.run(
                        [train_op, cross_entropy, summary_op, global_step], 
                        feed_dict={x: batch_x, y_: batch_y})
        writer.add_summary(summary, step)

        print(  " Epoch: %2d," % (epoch+1), 
                " Batch: %3d of %3d," % (i+1, batch_count), 
                " Cost: %.4f," % cost)

    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("Final Cost: %.4f" % cost)

  sv.stop()
  print("done")