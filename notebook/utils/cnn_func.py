import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt


def conv_model(X_train, Y_train, X_test, Y_test, config, **para_dict):
    '''
    Implement a three layer convolutional net in tensorflow
    
    Arg
    ---
    X_train, Y_train, X_test, Y_test: training and test sets and labels, with shape (None, 64, 64, 3), n_y= 6, 
    para_dict:                        dictionary of NN parameters
    
    Out
    ---
    
    
    '''
    lr = para_dict.get('learning_rate', 0.009)
    num_epochs = para_dict.get('num_epochs', 50)
    minibatch_size = para_dict.get('minibatch_size', 64)
    print_cost = para_dict.get('print_cost', True)
    batch_norm = para_dict.get('batch_norm', False)
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    
    X, Y = tf_placeholder(n_H0, n_W0, n_C0, n_y)
    parameters = init_parameters()
    Z3 = conv_propagation(X, parameters, batch_norm = batch_norm)
    cost = conv_cost(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1#??
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost/num_minibatches
            
            if print_cost == True and epoch %20 ==0:
                print('Cost after epoch %i: %f'% (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 ==0:
                costs.append(minibatch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate = '+ str(lr))
        plt.show()
    
        #predict_op = tf.argmax(Z3, 1)
        #correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float')) #return a accurarcy instance for evaluation
        accuracy = conv_accuracy(X, Y, Z3)
        train_acc = accuracy.eval({X: X_train, Y:Y_train})
        test_acc = accuracy.eval({X: X_test, Y:Y_test})
        print('Train Accuracy: ', train_acc)
        print('Test Accuracy: ', test_acc)
    
        return parameters

    
def conv_accuracy(X, Y, Z):
    '''
    helper function for evaluate training and testing accuracy
        
    '''
    predict_op = tf.argmax(Z, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float')) #return a accurarcy instance for evaluation
        
    return accuracy


def tf_placeholder(n_H0, n_W0, n_C0, n_y):
    '''
    function for initiate NN network
    
    '''
    
    X = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape= [None, n_y])
    
    
    return X, Y


def init_parameters():
    '''initial parameter for convolutional 2d network'''
    
    tf.set_random_seed(1)
    
    W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', [2, 2, 8, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameter_dict = {'W1': W1, 'W2': W2}
    
    return parameter_dict


def conv_propagation(X, parameters, batch_norm = True):
    '''
    function for forward propagation: CONV2D => RELU => MAXPOOL => CONV2D => RELU => MAXPOOL => FLATTEN => FULLYCONNECTED
    
    TODO
    ----
    implement batch normalization

    '''
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
    if batch_norm == True:
        Z1 = tf.layers.batch_normalization(Z1) #added one layer batch normalization
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding= 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
    if batch_norm == True:
        Z2 = tf.layers.batch_normalization(Z2)
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn = None)
    
    return Z3


#compute cost
def conv_cost(Z3, Y):
    '''
    function for compute cost of convolutional NN
    
    '''
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    cost = tf.reduce_mean(cost)
    
    return cost



#======== other utility functions ============


def load_dataset(DATA_PATH):
    train_dataset = h5py.File(DATA_PATH+'/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(DATA_PATH+'/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction
