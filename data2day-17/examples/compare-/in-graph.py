# in-graph replication
import tensorflow as tf
num_gpus = 2

# place the initial data on the cpu
with tf.device('/cpu:0'):
    input_data = tf.Variable([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.],
                              [10., 11., 12.]])
    b = tf.Variable([[1.], [1.], [2.]])

# split the data into chunks for each gpu
inputs = tf.split(input_data, num_gpus)
outputs = []

# loop over available gpus and pass input data
for i in range(num_gpus):
    with tf.device('/gpu:'+str(i)):
        outputs.append(tf.matmul(inputs[i], b))

# merge the results of the devices
with tf.device('/cpu:0'):
    output = tf.concat(outputs, axis=0)

# create a session and run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(output)