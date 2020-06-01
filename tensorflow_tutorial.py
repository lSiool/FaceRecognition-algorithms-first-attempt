
import tensorflow as tf
import tensorflow.python.framework.dtypes

print("Hello world")

#pip install numpy==1.16.4

#Граф это основа TensorFlow, все вычисления (операции), переменные находятся на графе. 
#Все, что происходит в коде, находится на дефолтном графе, предоставленном TensorFlow

graph = tf.get_default_graph() # доступ к этому графу 
#graph.get_operations() # список всех операций в этом графе




# 2 пособа начать сессию в тензорфлоу 

# sess=tf.Session()
# #... your code ...
# #... your code ...
# sess.close()
a=tf.constant(1.0)

b = tf.Variable(2.0,name="test_var")

init_op = tf.global_variables_initializer()

#Вы не можете распечатать или получить доступ к константе до тех пор пока не запустите сессию. 
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	print(sess.run(b))
# 	# Если вам необходимо напечатать название каждой операции
# 	for op in graph.get_operations():
# 		print(op.name)
#Плейсхолдеры — это тензоры, ожидающие инициализации данными
# a = tf.placeholder("float")
# b = tf.placeholder("float")
# y = tf.multiply(a, b)
# fd = {a: 2, b: 3}
# with tf.Session() as sess:
#     print(sess.run(y, fd))

import tensorflow as tf
import numpy as np

trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)

cost = (tf.pow(Y-y_model, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))