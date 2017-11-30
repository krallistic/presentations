"""Train a simple TF program to verify we can execute ops.
The program does a simple matrix multiplication.
Only the master assigns ops to devices/workers.
The master will assign ops to every task in the cluster. This way we can verify
that distributed training is working by executing ops on all devices.
"""
import argparse
import json
import logging
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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


def parse_args():
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--sleep_secs",
      default=0,
      type=int,
      help=("Amount of time to sleep at the end"))

  # TODO(jlewi): We ignore unknown arguments because the backend is currently
  # setting some flags to empty values like metadata path.
  args, _ = parser.parse_known_args()
  return args


def run(server, cluster_spec):  # pylint: disable=too-many-statements, too-many-locals
  """Build the graph and run the example.
  Args:
    server: The TensorFlow server to use.
  Raises:
    RuntimeError: If the expected log entries aren't found.
  """

  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # construct the graph and create a saver object
  with tf.Graph().as_default(): 
    # The initial value should be such that type is correctly inferred as
    # float.
    width = 10
    height = 10
    results = []
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):

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


    init_op = tf.global_variables_initializer()

    if server:
        target = server.target
    else:
      # Create a direct session.
        target = ""

    logging.info("Server target: %s", target)
    with tf.Session(
            target, config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init_op)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_count = int(mnist.train.num_examples/batch_size)
        for epoch in range(20):
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run( train_op, feed_dict={x: batch_x, y_: batch_y})
                if i % 10 == 0:
                    print("Batch: ", i, " from: " , batch_count)
                if epoch % 2 == 0: 
                    print("Epoch: ", epoch )
                    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("Done")
    print("Done")


def main():
  """Run training.
  Raises:
    ValueError: If the arguments are invalid.
  """
  logging.info("Tensorflow version: %s", tf.__version__)
  logging.info("Tensorflow git version: %s", tf.__git_version__)

  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  tf_config = json.loads(tf_config_json)
  logging.info("tf_config: %s", tf_config)

  task = tf_config.get("task", {})
  logging.info("task: %s", task)

  cluster_spec = tf_config.get("cluster", {})
  logging.info("cluster_spec: %s", cluster_spec)

  server = None
  device_func = None
  if cluster_spec:
    cluster_spec_object = tf.train.ClusterSpec(cluster_spec)
    server_def = tf.train.ServerDef(
        cluster=cluster_spec_object.as_cluster_def(),
        protocol="grpc",
        job_name=task["type"],
        task_index=task["index"])

    logging.info("server_def: %s", server_def)

    logging.info("Building server.")
    # Create and start a server for the local task.
    server = tf.train.Server(server_def)
    logging.info("Finished building server.")

    # Assigns ops to the local worker by default.
    device_func = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % server_def.task_index,
        cluster=server_def.cluster)
  else:
    # This should return a null op device setter since we are using
    # all the defaults.
    logging.error("Using default device function.")
    device_func = tf.train.replica_device_setter()

  job_type = task.get("type", "").lower()
  if job_type == "ps":
    logging.info("Running PS code.")
    server.join()
  elif job_type == "worker":
    logging.info("Running Worker code.")
    # The worker just blocks because we let the master assign all ops.
    server.join()
  elif job_type == "master" or not job_type:
    logging.info("Running master.")
    with tf.device(device_func):
      run(server=server, cluster_spec=cluster_spec)
  else:
    raise ValueError("invalid job_type %s" % (job_type,))


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  main()