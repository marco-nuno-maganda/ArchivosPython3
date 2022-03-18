#Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
#import tensorflow as tf
#tf.set_random_seed(seed=42)

import math

num_data = np.array([[0.5, 0.1], [0.3, 0.2], [0.7, 0.9],[0.8, 0.1]])
#(4, 2)
# array([[0.5, 0.1],
#        [0.3, 0.2],
#        [0.7, 0.9],
#        [0.8, 0.1]])

cat_data = np.array([[0], [1], [2], [0]])
#(4, 1)

one_hot_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
#(4, 3)

target = np.array([[0.1], [0.6], [0.4], [0.1]])
# (4, 1)

#def get_trainable_variables(graph=tf.get_default_graph()):
#    return [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

def miniBatch(x, y, batchSize):
    numObs  = x.shape[0]
    batches = [] 
    batchNum = math.floor(numObs / batchSize)
    
    if numObs % batchSize == 0:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
    else:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
        xBatch = x[batchNum * batchSize:, :]
        yBatch = y[batchNum * batchSize:, :]
        batches.append((xBatch, yBatch))
    return batches

data = np.concatenate((num_data, one_hot_cat), axis = 1)
#(4, 5)

n_features = data.shape[1]
n_outputs = target.shape[1]
n_hidden = 3

exit()

#tf.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('Placeholders'):
        X = tf.placeholder('float', shape=[None, n_features])
        #<tf.Tensor 'Placeholder:0' shape=(?, 5) dtype=float32>
        y = tf.placeholder('float', shape=[None, n_outputs])
        #<tf.Tensor 'Placeholder_1:0' shape=(?, 1) dtype=float32>

    with tf.name_scope("First_Layer"):
        W_fc1 = tf.get_variable('First_Layer/Hidden_layer_weights', initializer=tf.constant(np.array([[0.19, 0.55, 0.76],[0.33, 0.16, 0.97],[0.4 , 0.35, 0.7 ],[0.51, 0.85, 0.85],[0.54, 0.49, 0.57]]), dtype=tf.float32))
        #<tf.Variable 'First_Layer/Variable:0' shape=(5, 3) dtype=float32_ref>
        b_fc1 = tf.get_variable('First_Layer/Biases', initializer=tf.constant(np.array([0.1, 0.1, 0.1]), dtype=tf.float32))
        #<tf.Variable 'First_Layer/Variable_1:0' shape=(3,) dtype=float32_ref>
        h_fc1 = tf.nn.sigmoid(tf.matmul(X, W_fc1) + b_fc1)
        #<tf.Tensor 'First_Layer/Relu:0' shape=(?, 3) dtype=float32>

    with tf.name_scope("Output_Layer"):
        W_fc2 = tf.get_variable('Output_Layer/Output_layer_weights', initializer=tf.constant(np.array([[ 0.10],[ 0.03],[-0.17]]), dtype=tf.float32))
        # <tf.Variable 'Output_Layer/Variable:0' shape=(3, 1) dtype=float32_ref>
        b_fc2 = tf.get_variable('Output_Layer/Biases', initializer=tf.constant(np.array([0.1]), dtype=tf.float32))
        # <tf.Variable 'Output_Layer/Variable_1:0' shape=(1,) dtype=float32_ref>
        y_pred = tf.cast(tf.matmul(h_fc1, W_fc2) + b_fc2, dtype = tf.float32)
        #<tf.Tensor 'Output_Layer/add:0' shape=(?, 1) dtype=float32>

    with tf.name_scope("Loss"):
        loss = tf.losses.mean_squared_error(labels = y, predictions = y_pred)

    with tf.name_scope('Train'):
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        grads_and_vars = optimizer.compute_gradients(loss)
        trainer = optimizer.apply_gradients(grads_and_vars)

    # [(<tf.Tensor 'Train/gradients/First_Layer/MatMul_grad/tuple/control_dependency_1:0' shape=(5, 3) dtype=float32>,
    #   <tf.Variable 'First_Layer/Variable:0' shape=(5, 3) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/First_Layer/add_grad/tuple/control_dependency_1:0' shape=(3,) dtype=float32>,
    #   <tf.Variable 'First_Layer/Variable_1:0' shape=(3,) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/Output_Layer/MatMul_grad/tuple/control_dependency_1:0' shape=(3, 1) dtype=float32>,
    #   <tf.Variable 'Output_Layer/Variable:0' shape=(3, 1) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/Output_Layer/add_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>,
    #   <tf.Variable 'Output_Layer/Variable_1:0' shape=(1,) dtype=float32_ref>)]

    with tf.name_scope("Init"):
        global_variables_init = tf.global_variables_initializer()
        
get_trainable_variables(graph=graph)
# [<tf.Variable 'First_Layer/Hidden_layer_weights:0' shape=(5, 3) dtype=float32_ref>,
#  <tf.Variable 'First_Layer/Biases:0' shape=(3,) dtype=float32_ref>,
#  <tf.Variable 'Output_Layer/Output_layer_weights:0' shape=(3, 1) dtype=float32_ref>,
#  <tf.Variable 'Output_Layer/Biases:0' shape=(1,) dtype=float32_ref>]

with tf.Session(graph=graph) as sess:
    global_variables_init.run()
    tf.get_default_graph().finalize()
    print("Initialized")
    
    print ("Variables before training")
    old_var = {}
    for var in tf.global_variables():
        old_var[var.name] = sess.run(var)
        #print (var.name, sess.run(var))
    print(old_var)
    print('\n\n')
    
    miniBatches = miniBatch(data, target, batchSize = 1)
    total_batch = len(miniBatches) 
    i=1
    for batch in miniBatches:
        print('\n{}-observation\n'.format(i))
        xBatch = batch[0]
        yBatch = batch[1]
        _, loss_val, h_fc1_val, grads_and_vars_val, y_pred_val = sess.run([trainer, loss, h_fc1, grads_and_vars, y_pred], feed_dict={X: xBatch, y: yBatch})
        print('Loss: {}'.format(loss_val))
        print('Prediction: {}'.format(y_pred_val))
        print('Hidden layer forward prop:{}'.format(h_fc1_val))
        print('\n\n')
        print(grads_and_vars_val)
        i += 1
    print("Optimization Finished!")   
    print('\n\n')
    print ("Variables after training")
    new_var = {}
    for var in tf.global_variables():
        new_var[var.name] = sess.run(var)
    print(new_var)
        
# Initialized
# Variables before training
# {'First_Layer/Hidden_layer_weights:0': array([[0.19, 0.55, 0.76],
#        [0.33, 0.16, 0.97],
#        [0.4 , 0.35, 0.7 ],
#        [0.51, 0.85, 0.85],
#        [0.54, 0.49, 0.57]], dtype=float32), 'First_Layer/Biases:0': array([0.1, 0.1, 0.1], dtype=float32), 'Output_Layer/Output_layer_weights:0': array([[ 0.1 ],
#        [ 0.03],
#        [-0.17]], dtype=float32), 'Output_Layer/Biases:0': array([0.1], dtype=float32)}


