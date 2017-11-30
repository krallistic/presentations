import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None
batch_size = 50

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
  ps_hosts = FLAGS.ps_hosts.split(",")
  print(ps_hosts)
  worker_hosts = FLAGS.worker_hosts.split(",")
  print(worker_hosts)
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  # Create and start a server for the local task.
  server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    print("ps job, joining server")
    server.join()
  elif FLAGS.job_name == "worker":
    print("worker job, building graph")


    with tf.device("/jop:ps/task:0/cpu:0"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])


    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
        # target 10 output classes
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        # Build model...
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
        global_step = tf.contrib.framework.get_or_create_global_step()

        train_op = tf.contrib.layers.optimize_loss(
            loss=cross_entropy,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.01,
            # optimizer=tf.train.AdamOptimizer
            optimizer="Adam")
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=100)]
    init_op = tf.global_variables_initializer()

    
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=None,
                             saver=tf.train.Saver(),
                             global_step=global_step,
                             save_model_secs=600)
    with sv.managed_session(server.target) as mon_sess:
         for epoch in range(20):
             #Train  
             mon_sess.run(train_op)                   

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    print("Init Session")
    with sv.managed_session(server.target) as mon_sess:
    #with tf.train.MonitoredTrainingSession(master=server.target,
    #                                       is_chief=True,
    #                                       checkpoint_dir="/",
    #                                       hooks=hooks) as mon_sess:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
    #    mon_sess.run(init, feed_dict={x: batch_x, y_: batch_y})
        print("Done Init Variables")
        #while not mon_sess.should_stop():
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.
            # mon_sess.run handles AbortedError in case of preempted PS.

        batch_count = int(mnist.train.num_examples/batch_size)
        for epoch in range(20):
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                mon_sess.run( train_op, feed_dict={x: batch_x, y_: batch_y})
                if i % 10 == 0:
                    print("Batch: ", i, " from: " , batch_count)
                if epoch % 2 == 0: 
                    print("Epoch: ", epoch )
                    print("Test-Accuracy: %2.2f" % mon_sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("Done")
    print("Done")

if __name__ == "__main__":
  print("Starting...")
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
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
  print("Starting with Flags..: ", str(FLAGS))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)