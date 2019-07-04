#import 
from cnn_func import *
from utils.cnn_func import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def main():
    #get data Desktop/playground/deep_learning_comparative_study/exercises/data
    DATA_PATH = '/Users/ujv424/Desktop/playground/deep_learning_comparative_study/exercises/data'
    train_set_X_orig, train_set_Y_orig, test_set_X_orig, test_set_Y_orig, classes = load_dataset(DATA_PATH)
    #prepare data for model input
    X_train = train_set_X_orig/255.
    X_test = test_set_X_orig/255.
    Y_train = convert_to_one_hot(train_set_Y_orig, 6).T
    Y_test = convert_to_one_hot(test_set_Y_orig, 6).T

    para_dict = {'learning_rate': 0.01, 'num_epochs':100 , 'minibatch_size':64, 'print_cost':True }
    parameters = conv_model(X_train, Y_train, X_test, Y_test, config, **para_dict)


if __name__ == '__main__':
    main()