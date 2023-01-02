import tensorflow as tf

print("-"*50)

t1 = tf.constant([1, 1, 1])
t2 = tf.constant([2, 2, 2])
t3 = tf.constant([3, 3, 3])
print(tf.concat((t1, t2, t3), axis=0))
print(tf.stack((t1, t2, t3), axis=0))
print(tf.concat((tf.expand_dims(t1, 0), tf.expand_dims(t2, 0), tf.expand_dims(t3, 0)), axis=0))

print("-"*50)

print(tf.stack((t1, t2, t3), axis=1))
print(tf.concat((tf.expand_dims(t1, 1), tf.expand_dims(t2, 1), tf.expand_dims(t3, 1)), axis=1))








