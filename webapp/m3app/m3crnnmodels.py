import pandas as pd
import tensorflow as tf
import numpy as np

# only works on tensorflow version 1.1.0

global_config = {
    "lr": 1e-5,
    "batch_size": 12,
    "epochs": 10,
    "rotations": 30,
    "cnn_p": 0.9,
    "rnn_p": 0.9,
    "cnn_l2": 0.05,
    "rnn_l2": 0.05,
    "rnn_size": 256,
    "rnn_type": "gru",
    "bidirectional": True,
    "attention": False, # Make Attention Mechanisms Great Again (MAGMA :P)
    "alpha": 0.01, # not sure what this one does, was in your original code
    "rnn_multi": 1, #this one has not been implemented yet ... hoping to be done before you come back
}

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_conv_graph(state_size = 256, batch_size = 64, num_classes = 2, dim_size = 512):
    reset_graph()
    # Placeholders
    x = tf.placeholder(tf.float32, [batch_size, None, 256, 256, 1]) # [batch_size, num_steps] #N X T x D
    seqlen = tf.placeholder(tf.int32, [batch_size]) # this determines number of hidden unit layers yo!
    y = tf.placeholder(tf.int32, [batch_size])
    is_training = tf.placeholder(tf.bool)
    
    keep_prob = global_config["rnn_p"]
    rec_type = global_config["rnn_type"]
    
    regularizer, conv_wts, conv_bss, affine_wts, affine_bss = init_conv_layers_basic()
    regularizer = tf.contrib.layers.l2_regularizer(scale=global_config["rnn_l2"])
    
    videos = tf.unstack(x, axis=0)
    embeddings_mri = []
    firstnall = []
    for m in videos:
        embeddings, firstn = conv_embedding_lookup(m, conv_wts, conv_bss, affine_wts, affine_bss, regularizer, is_training) # no embedding lookup --- maybe update to do end-to-end training :)
        embeddings_mri.append(embeddings)
        firstnall.append(firstn)
    rnn_inputs = tf.stack(embeddings_mri, axis=0)
    
    print (tf.shape(rnn_inputs))       

    if global_config["bidirectional"]:
        if rec_type == "lstm":
            fw_cell =  tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
            bw_cell =  tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
        elif rec_type == "gru":
            fw_cell = tf.contrib.rnn.GRUCell(state_size)
            bw_cell = tf.contrib.rnn.GRUCell(state_size)
        init_state_fw = fw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        init_state_bw = bw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, rnn_inputs, sequence_length=seqlen, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
        last_rnn_output_fw = tf.gather_nd(rnn_outputs[0], tf.stack([tf.range(batch_size), seqlen-1], axis=1))
        last_rnn_output_bw = tf.gather_nd(rnn_outputs[1], tf.stack([tf.range(batch_size), seqlen-1], axis=1))
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, num_classes], regularizer=regularizer)
            W1 = tf.get_variable('W1', [state_size, num_classes], regularizer=regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(last_rnn_output_fw, W) + tf.matmul(last_rnn_output_bw, W1) + b
    else:
        if rec_type == "lstm":
            cell =  tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
        elif rec_type == "gru":
            cell = tf.contrib.rnn.GRUCell(state_size)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)
        last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seqlen-1], axis=1))
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, num_classes], regularizer=regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(last_rnn_output, W) + b
    
    scores = tf.nn.softmax(logits)
    preds = tf.cast(tf.argmax(scores,1), tf.int32)
    correct = tf.equal(preds, y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)) + reg_term
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(global_config["lr"]).minimize(loss)
    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'is_training': is_training,
        'dropout': keep_prob,
        'scores': scores,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy,
        'firstnall': firstnall
    }

def init_conv_layers_basic():
    regularizer = tf.contrib.layers.l2_regularizer(scale=global_config["cnn_l2"])
    Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 1, 8], regularizer=regularizer)
    bconv1 = tf.get_variable("bconv1", shape=[8]) 
    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 8, 8], regularizer=regularizer)
    bconv2 = tf.get_variable("bconv2", shape=[8])
    Wconv3 = tf.get_variable("Wconv3", shape=[5, 5, 8, 8], regularizer=regularizer)
    bconv3 = tf.get_variable("bconv3", shape=[8])
    Wconv4 = tf.get_variable("Wconv4", shape=[5, 5, 8, 8], regularizer=regularizer)
    bconv4 = tf.get_variable("bconv4", shape=[8])

    W_affine1024 = tf.get_variable("W_affine1024", shape=[800, 1024], regularizer=regularizer)
    b_affine1024 = tf.get_variable("b_affine1024", shape=[1024])
    W_affine512 = tf.get_variable("W_affine512", shape=[1024, 512], regularizer=regularizer)
    b_affine512 = tf.get_variable("b_affine512", shape=[512])
    
    conv_wts = (Wconv1, Wconv2, Wconv3, Wconv4)
    conv_bss = (bconv1, bconv2, bconv3, bconv4)
    affine_wts = (W_affine1024, W_affine512)
    affine_bss = (b_affine1024, b_affine512)
    return regularizer, conv_wts, conv_bss, affine_wts, affine_bss

def conv_embedding_lookup(X, conv_wts, conv_bss, affine_wts, affine_bss, regularizer, is_training):
    Wconv1, Wconv2, Wconv3, Wconv4 = conv_wts
    bconv1, bconv2, bconv3, bconv4 = conv_bss
    W_affine1024, W_affine512 = affine_wts
    b_affine1024, b_affine512 = affine_bss
    p = global_config["cnn_p"] # keep prob
    alpha = global_config["alpha"]

    cnn_f = {}
    out = tf.nn.avg_pool(X, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #starts 256 x 256. ends 128 x128
    
    # conv relu batchnorm pool
    out = tf.nn.conv2d(out, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1 #starts 128 by 128 by 1, ends 124 by 124 by 32
    cnn_f["CNN1FOVIZ"] = out
    out = tf.nn.relu(out)
    cnn_f["CNN1RELU"] = out
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = False)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #62 by 62 by 32

    out = tf.nn.conv2d(out, Wconv2, strides=[1,1,1,1], padding='VALID') + bconv2 #starts 62 by 62 by 32, ends 58 by 58 by 32
    cnn_f["CNN2FOVIZ"] = out
    out = tf.nn.relu(out)
    cnn_f["CNN2RELU"] = out
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = False)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #29 by 29 by 32
    
    out = tf.nn.conv2d(out, Wconv3, strides=[1,1,1,1], padding='VALID') + bconv3 #starts 29 by 29, ends 25 by 25 by 32
    cnn_f["CNN3FOVIZ"] = out
    out = tf.nn.relu(out)
    cnn_f["CNN3RELU"] = out
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = False)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    
    out = tf.nn.conv2d(out, Wconv4, strides=[1,1,1,1], padding='VALID') + bconv4 #starts 25 by 25, ends 21 by 21 by 32
    cnn_f["CNN4FOVIZ"] = out
    out = tf.nn.relu(out)
    cnn_f["CNN4RELU"] = out
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = False)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #10 by 10 by 32

    out = tf.reshape(out,[-1,800])
    
    out = tf.matmul(out,W_affine1024) + b_affine1024
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = False)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)

    out = tf.matmul(out,W_affine512) + b_affine512
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = False)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    
    return out, cnn_f