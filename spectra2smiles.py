"""Builds the spectra2smiles network.
"""

import tensorflow as tf
import numpy as np
import os

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# -----Main Function-----
TRAIN = False  # train a model else test data on previously created model

# -----Training Parameters-----
NUM_LAYERS = 2
NUM_TRAIN = 100000  # number of samples to be used for training with the rest being validation
LOAD_TRAINING_MODEL = False  # Continue training from previously saved model
BATCH_SIZE = 100
LEARN_RATE = 1.0e-3
L2_WEIGHT = 0
EPOCH_CHECK = 1  # How often to save model and record loss / accuarcy values
CLIP = 0.0  # Clips gradiants to avoid overflow (not active if equal to 0)
NOISE = 0  # The standard deviation of gaussian noise that is added to the gradiants during back propogation (not active if equal to 0)
NUM_THREADS = 8

# -----Convolution Network Parameters-----
SPECTRA_SIZE = 600
CONV_KEEP_PROB = 1.0  # Probability of keeping node during training batch (1.0 equals no dropout)
CONV_BATCH_NORM = True  # http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
KERNAL_SIZE = [25] * NUM_LAYERS
NUM_KERNALS = [100] * NUM_LAYERS
POOL = [2] * NUM_LAYERS
FIRST_LAYER_CONNECT = False  # See readme for explanation

# -----LSTM Network Parameters-----
LSTM_NUM_NODES = 100
SMILES_SIZE = 185  # The maximum length of any sample SMILES string
SMILES_VOCAB_SIZE = 17  # One extra spot for end of word signifier
SMILES_VOCAB = {0: "C", 1: "O", 2: "N", 3: "=", 4: "#", 5: "(", 6: ")",
                7: "1", 8: "2", 9: "3", 10: "4", 11: "5", 12: "6", 13: "7", 14: "8", 15: "9"}

# -----Parameters for Translation Network Between Convolution and LSTM networks-----
TRANSLATE_NUM_LAYERS = 0
TRANSLATE_NUM_NODES = 200  # only used if TRANSLATE_NUM_LAYERS > 0
TRANSLATE_BATCH_NORM_FLAG = True
TRANSLATE_KEEP_PROB = 1.0  # Probability of keeping node during training batch (1.0 equals no dropout)
DOUBLE_TRANSLATE = False  # See readme for explanation

# -----Save Path-----
TAG = "num_layers_" + str(NUM_LAYERS) + "_"
TAG = TAG + "num_conv_nodes_" + str(NUM_KERNALS[0]) + "_"
TAG = TAG + "kernal_size_" + str(KERNAL_SIZE[0]) + "_"
TAG = TAG + "num_lstm_nodes_" + str(LSTM_NUM_NODES) + "_"
if TRANSLATE_NUM_LAYERS > 0:
    TAG = TAG + "num_translate_layers_" + str(TRANSLATE_NUM_LAYERS) + "_"
    TAG = TAG + "num_translate_nodes_" + str(TRANSLATE_NUM_NODES) + "_"
TAG = TAG + "conv_dropout_" + str(CONV_KEEP_PROB) + "_"
TAG = TAG + "translate_dropout_" + str(TRANSLATE_KEEP_PROB) + "_"
TAG = TAG + "conv_norm_" + str(CONV_BATCH_NORM) + "_"
TAG = TAG + "translate_norm_" + str(TRANSLATE_BATCH_NORM_FLAG) + "_"
TAG = TAG + "L2_" + str(L2_WEIGHT) + "_"
TAG = TAG + "1stConnect_" + str(FIRST_LAYER_CONNECT) + "_"
TAG = TAG + "DT_" + str(DOUBLE_TRANSLATE) + "_"
TAG = TAG + "lr_" + str(LEARN_RATE) + "_elu/"

PATH_SAVE = "/Users/terryrabinowitz/Desktop/" + TAG
PATH_SAVE_TRAIN = PATH_SAVE + "train/"
if not os.path.exists(PATH_SAVE_TRAIN):
    os.makedirs(PATH_SAVE_TRAIN)
PATH_SAVE_VAL = PATH_SAVE + "val/"
if not os.path.exists(PATH_SAVE_VAL):
    os.makedirs(PATH_SAVE_VAL)

# -----Data Path-----
PATH_DATA = "/Users/terryrabinowitz/Desktop/compound_prediction/spectra2smiles/"
INPUT_PATH = PATH_DATA + "mz_shuffled_NEW.npy"  # shuffled spectra input
LABELS_PATH = PATH_DATA + "smiles_labels_shuffled_NEW.npy"  # shuffled smiles lables
INPUT_PATH_TEST = PATH_DATA + "mz_shuffled_test_NEW.npy"  # shuffled spectra input
LABELS_PATH_TEST = PATH_DATA + "smiles_labels_shuffled_test_NEW.npy"  # shuffled smiles lables


#######################################################################################################################

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
    # function to shuffle two data sets keeping corresponding pairs aligned
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def convolution_network(spectral_input, keep_prob_conv, norm_bool):
    """
    Args:
        spectral_input:     Tensor, 2D (number of samples x spectra size)
        keep_prob_conv:     Tensor, 1D (probability of keeping node during dropout)
        norm_bool:          boolean tf.Variable (used for convolution batch norm - True indicates training phase)
    Return:
        states:             A list containing all convolution output layers
    """
    states = []  # list to store output of each convolution layer to be used as state input to lstm network at the same level
    print spectral_input.get_shape()
    spectral_input = tf.reshape(spectral_input, shape=[-1, SPECTRA_SIZE, 1, 1])  # reshaped to fit convolution2d
    print spectral_input.get_shape()
    # first layer convolution
    kernal = tf.Variable(tf.truncated_normal(shape=[KERNAL_SIZE[0], 1, 1, NUM_KERNALS[0]], stddev=0.1),
                         name="conv_kernal_0")
    bias = tf.Variable(tf.zeros(shape=[NUM_KERNALS[0]]), name="conv_bias_0")
    state = tf.nn.conv2d(spectral_input, kernal, [1, 1, 1, 1], 'VALID')
    state = tf.nn.max_pool(state, [1, POOL[0], 1, 1], [1, POOL[0], 1, 1], 'VALID')

    if CONV_BATCH_NORM:
        state = batch_norm_conv(state, NUM_KERNALS[0], norm_bool)
    state = tf.nn.elu(tf.nn.bias_add(state, bias))
    state = tf.nn.dropout(state, keep_prob=keep_prob_conv)
    print state.get_shape()
    states.append(state)
    # second to final layer convolution (skipped if num_layers = 1)
    for layer_num in range(1, len(NUM_KERNALS)):
        name = "conv_kernal_" + str(layer_num)
        kernal = tf.Variable(tf.truncated_normal(
            shape=[KERNAL_SIZE[layer_num], 1, NUM_KERNALS[layer_num - 1], NUM_KERNALS[layer_num]], stddev=0.1),
            name=name)
        name = "conv__bias_" + str(layer_num)
        bias = tf.Variable(tf.zeros(shape=[NUM_KERNALS[layer_num]]), name=name)
        state = tf.nn.conv2d(state, kernal, [1, 1, 1, 1], 'VALID')
        state = tf.nn.max_pool(state, [1, POOL[layer_num], 1, 1], [1, POOL[layer_num], 1, 1], 'VALID')
        if CONV_BATCH_NORM:
            state = batch_norm_conv(state, NUM_KERNALS[layer_num], norm_bool)
        state = tf.nn.elu(tf.nn.bias_add(state, bias))
        state = tf.nn.dropout(state, keep_prob=keep_prob_conv)
        print state.get_shape()
        states.append(state)
    states = [tf.reshape(i, [-1, int(i.get_shape()[1]) * int(i.get_shape()[3])]) for i in states]
    if FIRST_LAYER_CONNECT:
        states = [tf.concat((i, states[0]), axis=1) for i in states]
    return states


def translation_network(conv_state, keep_prob_translate, tag):
    """
    Args:
        conv_state:              Tensor, 2D (number of samples x state size)
        keep_prob_translate:     Tensor, 1D (probability of keeping node during dropout)
        tag:                     naming string indicating whether state is for 'h' or 'c' of lstm and what level)
    Return:
        conv_state_translated:   Tensor, 2D (number of samples x state size)
    """
    print conv_state.get_shape()
    name = "translate_weightsA_" + tag
    if TRANSLATE_NUM_LAYERS > 0:
        num_nodes = TRANSLATE_NUM_NODES
    else:
        num_nodes = LSTM_NUM_NODES
    weights = tf.Variable(
        tf.truncated_normal(shape=[int(conv_state.get_shape()[1]), num_nodes], stddev=0.1), name=name)
    name = "translate_biases_" + tag
    biases = tf.Variable(tf.zeros([num_nodes]), name=name)
    conv_state_translated = tf.add(tf.matmul(conv_state, weights), biases)
    if TRANSLATE_BATCH_NORM_FLAG:
        batch_mean, batch_var = tf.nn.moments(conv_state_translated, [0])
        name = "translate_norm_scale_" + tag
        scale = tf.Variable(tf.ones([num_nodes]), name=name)
        name = "translate_norm_beta_" + tag
        beta = tf.Variable(tf.zeros([num_nodes]), name=name)
        conv_state_translated = tf.nn.batch_normalization(conv_state_translated, batch_mean, batch_var, beta, scale,
                                                          variance_epsilon=1e-3)
    conv_state_translated = tf.nn.dropout(conv_state_translated, keep_prob=keep_prob_translate)
    conv_state_translated = tf.nn.elu(conv_state_translated)
    print conv_state_translated.get_shape()

    for counter in range(TRANSLATE_NUM_LAYERS):
        if counter == TRANSLATE_NUM_LAYERS - 1:
            num_nodes = LSTM_NUM_NODES
        else:
            num_nodes = TRANSLATE_NUM_NODES
        name = "translate_weights" + str(counter) + "_" + tag
        weights = tf.Variable(tf.truncated_normal(shape=[TRANSLATE_NUM_NODES, num_nodes], stddev=0.1),
                              name=name)
        name = "translate_biases_" + tag
        biases = tf.Variable(tf.zeros([num_nodes]), name=name)
        conv_state_translated = tf.add(tf.matmul(conv_state_translated, weights), biases)
        if TRANSLATE_BATCH_NORM_FLAG:
            batch_mean, batch_var = tf.nn.moments(conv_state_translated, [0])
            name = "translate_norm_scale_" + tag
            scale = tf.Variable(tf.ones([num_nodes]), name=name)
            name = "translate_norm_beta_" + tag
            beta = tf.Variable(tf.zeros([num_nodes]), name=name)
            conv_state_translated = tf.nn.batch_normalization(conv_state_translated, batch_mean, batch_var, beta, scale,
                                                              variance_epsilon=1e-3)
        conv_state_translated = tf.nn.dropout(conv_state_translated, keep_prob=keep_prob_translate)
        conv_state_translated = tf.nn.elu(conv_state_translated)
        print conv_state_translated.get_shape()
    return conv_state_translated


def lstm_network(zero_input, conv_state_translated):
    """
    Args:
        zero_input:                Tensor, 3D (number of samples x smiles size x smiles vocabulary)
        conv_state_translated:     Tensor, 2D (number of samples x state size)
    Return:
        smiles_output:             Tensor, 3D (number of samples x smiles size x smiles vocabulary)
    """
    print zero_input.get_shape()
    cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.LSTMCell(LSTM_NUM_NODES, forget_bias=1.0) for _ in
         range(NUM_LAYERS)])
    smiles_output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=zero_input, initial_state=conv_state_translated)
    print smiles_output.get_shape()
    smiles_output = tf.reshape(smiles_output, [-1, LSTM_NUM_NODES])
    weights = tf.Variable(tf.truncated_normal([LSTM_NUM_NODES, SMILES_VOCAB_SIZE], stddev=0.1),
                          name="smiles_decode_weights")
    biases = tf.Variable(tf.zeros([SMILES_VOCAB_SIZE]), name="smiles_decode_biases")
    print smiles_output.get_shape()
    smiles_output = tf.matmul(smiles_output, weights) + biases
    print smiles_output.get_shape()
    smiles_output = tf.reshape(smiles_output, [-1, SMILES_SIZE, SMILES_VOCAB_SIZE])
    print smiles_output.get_shape()
    return smiles_output


def cost_function(pred, true):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=true)
    cost = tf.reduce_mean(cross_entropy)
    return cost


def accuracy_function(arg_max_true, arg_max_pred):
    end_of_smile_true = [np.nonzero(i == (SMILES_VOCAB_SIZE - 1))[0][0] for i in arg_max_true]
    end_of_smile_pred = []
    for i in arg_max_pred:
        try:
            end_of_smile_pred.append(np.nonzero(i == (SMILES_VOCAB_SIZE - 1))[0][0])
        except:
            end_of_smile_pred.append(len(i))
    soft_accuracies = []
    hard_accuracies = []
    for sample_num in xrange(len(arg_max_pred)):
        end_true = end_of_smile_true[sample_num]
        end_pred = end_of_smile_pred[sample_num]
        # accuracy_limit = max(end_true, end_pred)
        accuracy_limit = end_true
        pred_final = arg_max_pred[sample_num, :accuracy_limit + 1]
        true_final = arg_max_true[sample_num, :accuracy_limit + 1]
        num_matches = len([i for i, j in zip(pred_final, true_final) if i == j])
        soft_accuracy = float(num_matches) / len(pred_final)
        soft_accuracies.append(soft_accuracy)
        if num_matches == len(pred_final):
            hard_accuracies.append(1.0)
        else:
            hard_accuracies.append(0.0)
    return soft_accuracies, hard_accuracies


def train():
    print
    print("Loading Data")

    input = np.load(INPUT_PATH)
    labels = np.load(LABELS_PATH)
    input_train = input[:NUM_TRAIN]
    input_val = input[NUM_TRAIN:]
    labels_train = labels[:NUM_TRAIN]
    labels_val = labels[NUM_TRAIN:]
    print input_train.shape, input_val.shape
    print labels_train.shape, labels_val.shape

    print
    print("Initializing Tensor Variables")

    epoch = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    epoch_add_op = epoch.assign(epoch + 1)
    loaded_epoch = tf.placeholder(dtype=tf.float32)
    epoch_load_op = epoch.assign(loaded_epoch)
    norm_bool = tf.placeholder(dtype=tf.bool)
    learning_rate = tf.Variable(LEARN_RATE, trainable=False, dtype=tf.float32)
    spectra_x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, SPECTRA_SIZE])
    smiles_y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, SMILES_SIZE, SMILES_VOCAB_SIZE])
    smiles_zero = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, SMILES_SIZE, SMILES_VOCAB_SIZE])
    keep_prob_conv = tf.placeholder(dtype=tf.float32)
    keep_prob_translate = tf.placeholder(dtype=tf.float32)
    l2_param = tf.constant(L2_WEIGHT, dtype=tf.float32)
    argmax = tf.argmax(smiles_y, 2)

    print
    print("Making Model")

    conv_states = convolution_network(spectra_x, keep_prob_conv, norm_bool)
    conv_states_translated = []
    for layer_num in range(NUM_LAYERS):
        tag = "c" + str(layer_num)
        conv_state_translated1 = translation_network(conv_states[layer_num], keep_prob_translate, tag)
        if DOUBLE_TRANSLATE:
            tag = "h" + str(layer_num)
            conv_state_translated2 = translation_network(conv_states[layer_num], keep_prob_translate, tag)
        else:
            conv_state_translated2 = tf.zeros(shape=conv_state_translated1.get_shape())
        conv_states_translated.append(tf.contrib.rnn.LSTMStateTuple(conv_state_translated1, conv_state_translated2))
        print
    conv_states_translated = tuple(conv_states_translated)
    smiles_out = lstm_network(smiles_zero, conv_states_translated)

    total_parameters = 0
    l2 = 0
    counter = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if 'conv' not in variable.name:
            l2 += tf.nn.l2_loss(variable)
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        counter += 1
    print "Num Variables = ", counter, "\tTotal parameters = ", total_parameters

    cost = cost_function(smiles_out, smiles_y)
    cost_L2 = tf.multiply(l2_param, l2)
    cost = tf.add(cost, cost_L2)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)

    grads_and_vars = optimizer.compute_gradients(cost)
    if CLIP > 0:
        grads_and_vars = [(tf.clip_by_value(grad, -1 * CLIP, CLIP), var) for grad, var in grads_and_vars]
    if NOISE > 0:
        grads_and_vars = [(i[0] + tf.random_normal(shape=tf.shape(i[0]), mean=0, stddev=NOISE), i[1]) for i in
                          grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)

    print
    print("Running Session")

    saver = tf.train.Saver()
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                            inter_op_parallelism_threads=NUM_THREADS,
                            allow_soft_placement=True,
                            device_count={'CPU': NUM_THREADS})
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
            print epoch.eval(session=sess)
            shuffle_in_unison(labels_train, input_train)
            accuracy_total_hard = 0
            accuracy_total_soft = 0
            loss_total = 0
            counter = 0
            print 'loss', '\t', 'l2_loss', '\t', 'hard_accuracy', '\t', 'soft_accuracy'
            for batch_num in xrange(0, len(labels_train), BATCH_SIZE):
                train_batch_mz = input_train[batch_num:batch_num + BATCH_SIZE]
                train_batch_smiles = labels_train[batch_num:batch_num + BATCH_SIZE]
                train_batch_smiles_zeros = np.zeros(shape=train_batch_smiles.shape)
                _, smiles_output_train, loss, loss_l2 = sess.run(
                    [train_op, smiles_out, cost, cost_L2],
                    feed_dict={spectra_x: train_batch_mz,
                               smiles_y: train_batch_smiles,
                               smiles_zero: train_batch_smiles_zeros,
                               keep_prob_conv: CONV_KEEP_PROB,
                               keep_prob_translate: TRANSLATE_KEEP_PROB,
                               norm_bool: True})

                arg_max_true = sess.run(argmax, feed_dict={smiles_y: train_batch_smiles})
                arg_max_pred = sess.run(argmax, feed_dict={smiles_y: smiles_output_train})
                loss_total += loss
                soft_accuracies, hard_accuracies = accuracy_function(arg_max_true, arg_max_pred)
                accuracy_total_hard += np.mean(hard_accuracies)
                accuracy_total_soft += np.mean(soft_accuracies)
                counter += 1
                print loss, loss_l2, np.mean(hard_accuracies), np.mean(soft_accuracies)
            accuracy_hard_train = accuracy_total_hard / counter
            accuracy_soft_train = accuracy_total_soft / counter
            train_loss = loss_total / counter

            accuracy_total_hard = 0
            accuracy_total_soft = 0
            loss_total = 0
            counter = 0

            for batch_num in xrange(0, len(labels_val), BATCH_SIZE):
                if batch_num + BATCH_SIZE > len(labels_val):
                    break
                val_batch_mz = input_val[batch_num:batch_num + BATCH_SIZE]
                val_batch_smiles = labels_val[batch_num:batch_num + BATCH_SIZE]
                val_batch_smiles_zeros = np.zeros(shape=val_batch_smiles.shape)

                smiles_output_val, loss, loss_l2 = sess.run(
                    [smiles_out, cost, cost_L2],
                    feed_dict={spectra_x: val_batch_mz,
                               smiles_y: val_batch_smiles,
                               smiles_zero: val_batch_smiles_zeros,
                               keep_prob_conv: 1.0,
                               keep_prob_translate: 1.0,
                               norm_bool: False})

                arg_max_true = sess.run(argmax, feed_dict={smiles_y: val_batch_smiles})
                arg_max_pred = sess.run(argmax, feed_dict={smiles_y: smiles_output_val})
                loss_total += loss
                soft_accuracies, hard_accuracies = accuracy_function(arg_max_true, arg_max_pred)
                accuracy_total_hard += np.mean(hard_accuracies)
                accuracy_total_soft += np.mean(soft_accuracies)
                counter += 1
            accuracy_hard_val = accuracy_total_hard / counter
            accuracy_soft_val = accuracy_total_soft / counter
            val_loss = loss_total / counter

            tag = PATH_SAVE_TRAIN + "current_train_model"
            saver.save(sess, tag)

            if val_loss < previous_best:
                previous_best = val_loss
                tag = PATH_SAVE_VAL + "best_val_model"
                saver.save(sess, tag)

            if epoch.eval(session=sess) % EPOCH_CHECK == 0:
                tag = PATH_SAVE + "loss_summary.txt"
                string = str(train_loss) + " " + str(accuracy_hard_train) + " " + str(
                    accuracy_soft_train) + " " + str(
                    val_loss) + " " + str(accuracy_hard_val) + " " + str(accuracy_soft_val) + "\n"
                with open(tag, "a") as myfile:
                    myfile.write(string)

            sess.run(epoch_add_op)


def test():
    print
    print("Loading Data")

    input_test = np.load(INPUT_PATH_TEST)
    labels_test = np.load(LABELS_PATH_TEST)
    print input_test.shape, labels_test.shape

    print
    print("Initializing Tensor Variables")

    norm_bool = tf.placeholder(dtype=tf.bool)
    spectra_x = tf.placeholder(dtype=tf.float32, shape=[1, SPECTRA_SIZE])
    smiles_y = tf.placeholder(dtype=tf.float32, shape=[1, SMILES_SIZE, SMILES_VOCAB_SIZE])
    smiles_zero = tf.placeholder(dtype=tf.float32, shape=[1, SMILES_SIZE, SMILES_VOCAB_SIZE])
    keep_prob_conv = tf.placeholder(dtype=tf.float32)
    keep_prob_translate = tf.placeholder(dtype=tf.float32)
    argmax = tf.argmax(smiles_y, 2)

    print
    print("Making Model")

    conv_states = convolution_network(spectra_x, keep_prob_conv, norm_bool)
    conv_states_translated = []
    for layer_num in range(NUM_LAYERS):
        tag = "c" + str(layer_num)
        conv_state_translated1 = translation_network(conv_states[layer_num], keep_prob_translate, tag)
        if DOUBLE_TRANSLATE:
            tag = "h" + str(layer_num)
            conv_state_translated2 = translation_network(conv_states[layer_num], keep_prob_translate, tag)
        else:
            conv_state_translated2 = tf.zeros(shape=conv_state_translated1.get_shape())
        conv_states_translated.append(tf.contrib.rnn.LSTMStateTuple(conv_state_translated1, conv_state_translated2))
        print
    conv_states_translated = tuple(conv_states_translated)
    smiles_out = lstm_network(smiles_zero, conv_states_translated)

    total_parameters = 0
    counter = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        counter += 1
    print "Num Variables = ", counter, "\tTotal parameters = ", total_parameters

    saver = tf.train.Saver()
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                            inter_op_parallelism_threads=NUM_THREADS,
                            allow_soft_placement=True,
                            device_count={'CPU': NUM_THREADS})
    with tf.Session(config=config) as sess:
        print("Reading model parameters from %s" % PATH_SAVE_VAL)
        saver.restore(sess, tf.train.latest_checkpoint(PATH_SAVE_VAL))

        accuracy_total_hard = 0
        accuracy_total_soft = 0
        print 'sample number', '\t', 'hard_accuracy', '\t', 'soft_accuracy'
        for sample_num in xrange(len(labels_test)):
            test_mz = input_test[sample_num]
            test_mz = np.reshape(test_mz, (1, SPECTRA_SIZE))
            test_smiles = labels_test[sample_num]
            test_smiles = np.reshape(test_smiles, (1, SMILES_SIZE, SMILES_VOCAB_SIZE))
            test_smiles_zeros = np.zeros(shape=test_smiles.shape)
            smiles_output_test = sess.run(
                [smiles_out],
                feed_dict={spectra_x: test_mz,
                           smiles_y: test_smiles,
                           smiles_zero: test_smiles_zeros,
                           keep_prob_conv: 1.0,
                           keep_prob_translate: 1.0,
                           norm_bool: False})

            arg_max_true = sess.run(argmax, feed_dict={smiles_y: test_smiles})
            arg_max_pred = sess.run(argmax, feed_dict={smiles_y: smiles_output_test})

            soft_accuracy, hard_accuracy = accuracy_function(arg_max_true, arg_max_pred)
            accuracy_total_hard += hard_accuracy[0]
            accuracy_total_soft += soft_accuracy[0]
            print sample_num, hard_accuracy[0], soft_accuracy[0]
        accuracy_hard_test = accuracy_total_hard / len(labels_test)
        accuracy_soft_test = accuracy_total_soft / len(labels_test)
        print
        print 'average_hard_accuracy', '\t', 'average_soft_accuracy'
        print accuracy_hard_test, accuracy_soft_test


def main():
    if TRAIN:
        train()
    else:
        test()


###########################

main()
