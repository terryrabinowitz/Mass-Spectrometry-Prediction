"""Builds the spectra2formula network.
"""

import tensorflow as tf
import numpy as np
import os

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# -----Main Function-----
TRAIN = False  # train a model if True / test the model if False

# -----Training Parameters-----
NUM_TRAIN = 100000  # number of samples to be used for training with the rest being validation
LOAD_TRAINING_MODEL = False  # Continue training from previously saved model
BATCH_SIZE = 100
LEARN_RATE = 1.0e-2
L2_WEIGHT = 0 # L2 regularization weight
EPOCH_CHECK = 1  # How often to save model and record loss / accuarcy values
CLIP = 0.0  # Clips gradiants to avoid overflow (not active if equal to 0)
NOISE = 0  # The standard deviation of gaussian noise that is added to the gradiants during back propogation (not active if equal to 0)
NUM_THREADS = 8

# -----Convolution Subnetwork Parameters-----
SPECTRA_SIZE = 600 # width of spectral input
NUM_LAYERS_CONV = 2
MZ_KERNAL_SIZE = [25] * NUM_LAYERS_CONV
MZ_NUM_KERNALS = [100] * NUM_LAYERS_CONV
KEEP_PROB_CONV = 1.0  # Dropout in the convolution layers (1.0 = no dropout)
CONV_BATCH_NORM_FLAG = True

# -----Dense Subnetwork Parameters-----
NUM_LAYERS_DENSE = 2
MZ_NUM_NODES = [100] * NUM_LAYERS_DENSE
DENSE_BATCH_NORM_FLAG = True
KEEP_PROB_DENSE = 1.0  # Dropout in the dense layers (1.0 = no dropout)
NUM_ELEMENTS = 4 # Number of classes in the output labels (carbon, hydrogen, nitrogen, oxygen)

# -----Save Path-----
TAG = "num_layers_conv_" + str(NUM_LAYERS_CONV) + "_"
TAG = TAG + "num_conv_nodes_" + str(MZ_NUM_KERNALS[0]) + "_"
TAG = TAG + "kernal_size_" + str(MZ_KERNAL_SIZE[0]) + "_"
TAG = TAG + "num_layers_dense_" + str(NUM_LAYERS_DENSE) + "_"
TAG = TAG + "num_dense_nodes_" + str(MZ_NUM_NODES[0]) + "_"
TAG = TAG + "conv_dropout_" + str(KEEP_PROB_CONV) + "_"
TAG = TAG + "dense_dropout_" + str(KEEP_PROB_DENSE) + "_"
TAG = TAG + "conv_norm_" + str(CONV_BATCH_NORM_FLAG) + "_"
TAG = TAG + "dense_norm_" + str(DENSE_BATCH_NORM_FLAG) + "_"
TAG = TAG + "L2_" + str(L2_WEIGHT) + "_"
TAG = TAG + "noise_" + str(NOISE) + "_"
TAG = TAG + "lr_" + str(LEARN_RATE) + "/"

PATH_SAVE = "/Users/terryrabinowitz/Desktop/compound_prediction/spectra2formula/save/" + TAG

PATH_SAVE_TRAIN = PATH_SAVE + "train/"
if not os.path.exists(PATH_SAVE_TRAIN):
    os.makedirs(PATH_SAVE_TRAIN)
PATH_SAVE_VAL = PATH_SAVE + "val/"
if not os.path.exists(PATH_SAVE_VAL):
    os.makedirs(PATH_SAVE_VAL)

# -----Data Path-----
PATH_DATA = "/Users/terryrabinowitz/Desktop/compound_prediction/spectra2formula/"
LABEL_PATH = PATH_DATA + "formula_shuffled_NEW.npy"
INPUT_PATH = PATH_DATA + "mz_shuffled_NEW.npy"
LABEL_PATH_TEST = PATH_DATA + "formula_shuffled_test_NEW.npy"
INPUT_PATH_TEST = PATH_DATA + "mz_shuffled_test_NEW.npy"


####################################################


def batch_norm_conv(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def cost_function_sse(pred, true):
    cost = tf.nn.l2_loss(pred - true)
    return cost


def model(mz_input, keep_prob_conv, keep_prob_dense):
    stride = [1, 1, 1, 1]
    padding = 'VALID'
    print mz_input.get_shape()
    mz_input = tf.reshape(mz_input, shape=[-1, SPECTRA_SIZE, 1, 1])
    print mz_input.get_shape()
    kernal_encode = tf.Variable(tf.truncated_normal(shape=[MZ_KERNAL_SIZE[0], 1, 1, MZ_NUM_KERNALS[0]], stddev=0.1),
                                name="conv_kernal_0")
    bias_encode = tf.Variable(tf.zeros(shape=[MZ_NUM_KERNALS[0]]), name="conv_bias_0")
    state = tf.nn.conv2d(mz_input, kernal_encode, stride, padding)
    state = tf.nn.max_pool(state, [1, 2, 1, 1], [1, 2, 1, 1], padding)
    print state.get_shape()
    state = tf.nn.elu(tf.nn.bias_add(state, bias_encode))
    state = tf.nn.dropout(state, keep_prob=keep_prob_conv)
    for layer_num in range(1, len(MZ_NUM_KERNALS)):
        name = "conv_kernal_" + str(layer_num)
        kernal_encode = tf.Variable(tf.truncated_normal(
            shape=[MZ_KERNAL_SIZE[layer_num], 1, MZ_NUM_KERNALS[layer_num - 1], MZ_NUM_KERNALS[layer_num]], stddev=0.1),
            name=name)
        name = "conv__bias_" + str(layer_num)
        bias_encode = tf.Variable(tf.zeros(shape=[MZ_NUM_KERNALS[layer_num]]), name=name)
        state = tf.nn.conv2d(state, kernal_encode, stride, padding)
        state = tf.nn.max_pool(state, [1, 2, 1, 1], [1, 2, 1, 1], padding)
        state = tf.nn.elu(tf.nn.bias_add(state, bias_encode))
        state = tf.nn.dropout(state, keep_prob=keep_prob_conv)
        print state.get_shape()
    state = tf.reshape(state, [-1, int(state.get_shape()[1]) * int(state.get_shape()[3])])
    print state.get_shape()

    for layer_num in range(NUM_LAYERS_DENSE):
        if layer_num == 0:
            fan_in = int(state.get_shape()[1])
        else:
            fan_in = MZ_NUM_NODES[layer_num - 1]
        fan_out = MZ_NUM_NODES[layer_num]
        tag = "dense_weight_" + str(layer_num)
        weight = tf.Variable(tf.truncated_normal(shape=[fan_in, fan_out], stddev=0.1), name=tag)
        tag = "dense_bias_" + str(layer_num)
        bias = tf.Variable(tf.zeros([fan_out]), name=tag)
        state = tf.add(tf.matmul(state, weight), bias)
        if DENSE_BATCH_NORM_FLAG:
            batch_mean, batch_var = tf.nn.moments(state, [0])
            tag = "dense_norm_scale_" + str(layer_num)
            scale = tf.Variable(tf.ones([fan_out]), name=tag)
            tag = "dense_norm_beta_" + str(layer_num)
            beta = tf.Variable(tf.zeros([fan_out]), name=tag)
            state = tf.nn.batch_normalization(state, batch_mean, batch_var, beta, scale, variance_epsilon=1e-3)
        state = tf.nn.dropout(state, keep_prob=keep_prob_dense)
        state = tf.nn.elu(state)
        print state.get_shape()
    tag1 = "final_weight"
    tag2 = "final_bias"
    fan_in = int(state.get_shape()[1])
    fan_out = NUM_ELEMENTS
    weight = tf.Variable(
        tf.truncated_normal(shape=[fan_in, fan_out], stddev=0.1), name=tag1)
    bias = tf.Variable(tf.zeros([fan_out]), name=tag2)
    output = tf.add(tf.matmul(state, weight), bias)
    print output.get_shape()
    output = tf.nn.relu(output)
    return output


def train():
    print
    print("Loading Data")

    formula = np.load(LABEL_PATH)
    mz = np.load(INPUT_PATH)

    mz_train = mz[:NUM_TRAIN]
    mz_val = mz[NUM_TRAIN:]
    formula_train = formula[:NUM_TRAIN]
    formula_val = formula[NUM_TRAIN:]

    print mz_train.shape, mz_val.shape
    print formula_train.shape, formula_val.shape

    epoch = tf.Variable(1, trainable=False, dtype=tf.int32)
    epoch_add_op = epoch.assign(epoch + 1)
    loaded_epoch = tf.placeholder(dtype=tf.int32)
    epoch_load_op = epoch.assign(loaded_epoch)

    learning_rate = tf.Variable(LEARN_RATE, trainable=False, dtype=tf.float32)
    mz_x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, SPECTRA_SIZE])
    formula_x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_ELEMENTS])

    keep_prob_conv = tf.placeholder(dtype=tf.float32)
    keep_prob_dense = tf.placeholder(dtype=tf.float32)

    L2_param = tf.constant(L2_WEIGHT, dtype=tf.float32)

    formula_pred = model(mz_x, keep_prob_conv, keep_prob_dense)

    total_parameters = 0
    L2_cost = 0
    counter = 0
    print
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        L2_cost += tf.nn.l2_loss(variable)
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        counter += 1
    print
    print "Num Variables = ", counter, "\tTotal parameters = ", total_parameters

    cost = cost_function_sse(formula_pred, formula_x)
    L2_cost = tf.multiply(L2_param, L2_cost)
    cost = tf.add(cost, L2_cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1.0e-8)
    grads_and_vars = optimizer.compute_gradients(cost)
    if CLIP > 0:
        grads_and_vars = [(tf.clip_by_value(grad, -1 * CLIP, CLIP), var) for grad, var in grads_and_vars]
    if NOISE > 0:
        grads_and_vars = [(i[0] + tf.random_normal(shape=tf.shape(i[0]), mean=0, stddev=noise), i[1]) for i in
                          grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)

    formula_pred_int = tf.cast(formula_pred, tf.int32)
    formula_true_int = tf.cast(formula_x, tf.int32)
    matches = tf.reduce_sum(tf.cast(tf.equal(formula_pred_int, formula_true_int), tf.float32), axis=1)
    perfect = tf.reduce_sum(tf.convert_to_tensor(np.ones(formula_x.get_shape()), dtype=tf.float32), axis=1)
    accuracy_hard = tf.reduce_mean(tf.cast(tf.equal(matches, perfect), tf.float32))

    saver = tf.train.Saver()
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                            inter_op_parallelism_threads=NUM_THREADS,
                            allow_soft_placement=True,
                            device_count={'CPU': NUM_THREADS})
    print
    print("Running Session")
    with tf.Session(config=config) as sess:

        if LOAD_TRAINING_MODEL:
            print("Reading model parameters from %s" % PATH_SAVE_TRAIN)
            saver.restore(sess, tf.train.latest_checkpoint(PATH_SAVE_TRAIN))
            tag = PATH_SAVE + "loss_summary.txt"
            loss_file = open(tag)
            previous_best = np.inf
            counter = 1
            for line in loss_file:
                line = line.split()
                lv = float(line[3])
                if lv < previous_best:
                    previous_best = lv
                counter += 1
            loss_file.close()
            sess.run(epoch_load_op, feed_dict={loaded_epoch: counter})
            print "current epoch =", epoch.eval(session=sess)
            print "best validation error =", previous_best
        else:
            sess.run(tf.global_variables_initializer())
            previous_best = np.inf
        while True:
            shuffle_in_unison(formula_train, mz_train)
            loss_train = 0
            accuracy_train_hard = 0
            counter = 0
            for batch_num in xrange(0, len(formula_train), BATCH_SIZE):
                train_batch_mz = mz_train[batch_num:batch_num + BATCH_SIZE]
                train_batch_formula = formula_train[batch_num:batch_num + BATCH_SIZE]
                _, loss, loss_l2, accH, formula_out_train = sess.run(
                    [train_op, cost, L2_cost, accuracy_hard, formula_pred],
                    feed_dict={formula_x: train_batch_formula,
                               mz_x: train_batch_mz,
                               keep_prob_conv: KEEP_PROB_CONV,
                               keep_prob_dense: KEEP_PROB_DENSE})
                accuracy_train_hard += accH
                print loss, loss_l2, accH
                loss_train += loss
                counter += 1
            accuracy_train_hard /= counter
            accuracy_train_soft /= counter
            loss_train /= counter
            loss_test = 0
            accuracy_test_hard = 0
            counter = 0
            for batch_num in xrange(0, len(formula_val), BATCH_SIZE):
                test_batch_mz = mz_val[batch_num:batch_num + BATCH_SIZE]
                test_batch_formula = formula_val[batch_num:batch_num + BATCH_SIZE]
                if batch_num + BATCH_SIZE > len(formula_val):
                    break
                loss, loss_l2, accH, formula_out_test = sess.run(
                    [cost, L2_cost, accuracy_hard, formula_pred],
                    feed_dict={formula_x: test_batch_formula,
                               mz_x: test_batch_mz,
                               keep_prob_conv: 1.0,
                               keep_prob_dense: 1.0})
                accuracy_test_hard += accH

                loss_test += loss
                counter += 1
            accuracy_test_hard /= counter
            loss_test /= counter

            print loss_train, loss_test
            print accuracy_train_hard, accuracy_test_hard
            print
            tag = PATH_SAVE_TRAIN + "best_train_loss_model"
            saver.save(sess, tag)
            if loss_test < previous_best:
                previous_best = loss_test
                tag = PATH_SAVE_VAL + "best_val_loss_model"
                saver.save(sess, tag)
            if epoch.eval(session=sess) % EPOCH_CHECK == 0:
                tag = PATH_SAVE + "loss_summary.txt"
                string = str(loss_train) + " " + str(accuracy_train_hard) + " " + str(
                    loss_test) + " " + str(accuracy_test_hard) + "\n"
                with open(tag, "a") as myfile:
                    myfile.write(string)
            sess.run(epoch_add_op)


def test():
    print
    print("Loading Data")
    input_test = np.load(INPUT_PATH_TEST)
    labels_test = np.load(LABEL_PATH_TEST)

    mz_x = tf.placeholder(dtype=tf.float32, shape=[1, SPECTRA_SIZE])
    formula_x = tf.placeholder(dtype=tf.float32, shape=[1, NUM_ELEMENTS])

    keep_prob_conv = tf.placeholder(dtype=tf.float32)
    keep_prob_dense = tf.placeholder(dtype=tf.float32)

    formula_pred = model(mz_x, keep_prob_conv, keep_prob_dense)
    formula_pred_int = tf.cast(formula_pred, tf.int32)
    formula_true_int = tf.cast(formula_x, tf.int32)
    matches = tf.reduce_sum(tf.cast(tf.equal(formula_pred_int, formula_true_int), tf.float32), axis=1)
    perfect = tf.reduce_sum(tf.convert_to_tensor(np.ones(formula_x.get_shape()), dtype=tf.float32), axis=1)
    accuracy_hard = tf.reduce_mean(tf.cast(tf.equal(matches, perfect), tf.float32))
    total_parameters = 0
    counter = 0
    print
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        counter += 1
    print
    print "Num Variables = ", counter, "\tTotal parameters = ", total_parameters
    
    saver = tf.train.Saver()
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                            inter_op_parallelism_threads=NUM_THREADS,
                            allow_soft_placement=True,
                            device_count={'CPU': NUM_THREADS})
    with tf.Session(config=config) as sess:
        print("Reading model parameters from %s" % PATH_SAVE_VAL)
        saver.restore(sess, tf.train.latest_checkpoint(PATH_SAVE_VAL))
        
        accuracy_train_hard = 0
        for sample_num in xrange(len(labels_test)):
            test_mz = input_test[sample_num]
            test_mz = np.reshape(test_mz, (1, SPECTRA_SIZE))
            test_formula = labels_test[sample_num]
            test_formula = np.reshape(test_formula, (1, NUM_ELEMENTS))
            accH, formula_out_test = sess.run(
                [accuracy_hard, formula_pred],
                feed_dict={formula_x: test_formula,
                           mz_x: test_mz,
                           keep_prob_conv: 1.0,
                           keep_prob_dense: 1.0})
            accuracy_train_hard += accH
            print sample_num, accH
        print
        accuracy_train_hard /= len(labels_test)
        print accuracy_train_hard
    
    

######################################################

def main():
    if TRAIN:
        train()
    else:
        test()


main()
