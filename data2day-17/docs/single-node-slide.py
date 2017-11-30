import tensorflow as tf

num_workers = 2
a_large_number = 10000

def optimze_loss(a,b):
    return ""

if __name__ == '__main__':
    y_ = ""
    
    with tf.device("/cpu:0"):
        W = tf.Variable()
        b = tf.Variable()
    with tf.device("/gpu:0"):
        y = tf.matmul(input, W) + b
        train_op = optimze_loss(y, y_) 

    with tf.device("/job:ps/task:0/cpu:0"):
        W = tf.Variable()
        b = tf.Variable()
    y_split = tf.split(0, num_workers, y)
    for i in range(num_workers):
        with tf.device("job:worker/task:%d/gpu:0" % i):
            y = tf.matmul(y_split[i], W) + b
    train_op = optimze_loss(y, y_)    

    with tf.device("/jop:ps/task:0/cpu:0"):
        W = tf.Variable()
        b = tf.Variable()
    with tf.device("/job:worker/task:0/gpu:0"):
        y = tf.matmul(input, W) + b
        train_op = optimze_loss(y, y_) 

    with tf.device("/jop:ps/task:0/cpu:0"):
        W = tf.Variable()
        b = tf.Variable()
    with tf.device("/job:worker/task:1/gpu:0"):
        y = tf.matmul(input, W) + b
        train_op = optimze_loss(y, y_) 


    with tf.device("/jop:ps/task:0/cpu:0"):
        W_0 = tf.Variable()
        b_0 = tf.Variable()
    with tf.device("/jop:ps/task:1/cpu:0"):
        W_1 = tf.Variable()
        b_1 = tf.Variable()
    with tf.device("/job:worker/task:1/gpu:0"):
        y_1 = tf.matmul(input, W_0) + b_0
        #compute intesive Part
    with tf.device("/job:worker/task:2/gpu:0"):
        y = tf.matmul(y_1, W_1) + b_1
        #compute intensive Part
        train_op = optimze_loss(y, y_) 
    
    with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
        pass
        
    embedding = tf.get_variables("embedding", [a_large_number, 20],
         partitioner=tf.fixed_size_partitioner(3))