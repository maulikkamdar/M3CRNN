import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
import cv2
import sys

# editing all configurations in one place is the new sexy :p
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


class BucketedDataIterator():
    # update by reading the matrices from a disk ...
    def __init__(self, df, num_buckets = 4, dim=512):
        df = df.sort_values(seq_identifer).reset_index(drop=True)
        self.size = len(df) / num_buckets
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.iloc[int(bucket*self.size): int((bucket+1)*self.size)]) #not sure why to add -1 , because ix looks at length 
        self.num_buckets = num_buckets
        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()
        self.epochs = 0
        self.D = dim
        self.cur_bucket = 0
        
    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
            
    def next_batch_imgs(self, n):
        if self.cur_bucket == self.num_buckets-1 and self.cursor[self.cur_bucket] + n > self.size:
            self.epochs += 1
            self.shuffle()
            self.cur_bucket = 0
        elif self.cursor[self.cur_bucket] + n > self.size:
            self.cur_bucket += 1
        i = self.cur_bucket
        res = self.dfs[i].iloc[self.cursor[i]:self.cursor[i]+n]
        act_n = res.shape[0]
        self.cursor[i] += act_n
        maxlen = max(res[seq_identifer])
        x = np.zeros([act_n, maxlen, 256, 256, 1], dtype=np.float32)
        res = load_numpy_matrices_imgs_act(res, maxlen)
        for i in range(act_n): 
            x[i, 0:maxlen, :, :, :] += res["vectors"].values[i] #assuming res["vectors"] already holds TxD numpy matrices
            print x.shape
        return x, res[output_identifer], res[seq_identifer], res["shahash"] # x has a shape N X T X 256 x 256 x 1

def uniform_seq_gen(curr, shap_d = 110):
    if True:#curr.shape[0] > 100 and curr.shape[0] < 200:
        if curr.shape[0] < shap_d:
            remaining = shap_d-curr.shape[0]
            left_pad = int(remaining/2)
            right_pad = remaining - left_pad
            if left_pad > 0: 
                left = np.zeros((left_pad,256,256))
                curr = np.append(left, curr, axis = 0)
            if right_pad > 0:
                right = np.zeros((right_pad,256,256))
                curr = np.append(curr, right, axis = 0)
        elif curr.shape[0] > shap_d:
            start = int(len(curr)/2)-shap_d/2
            end = start + shap_d
            curr = curr[start:end,:,:]
    #print(curr.shape)
    return curr

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape)
    return result

def load_numpy_matrices_imgs_act(df, maxlen):
    def _load_matrix(x):
        #print(x)
        x_split = x.split("_") #shahash_rot_rev
        if len(x_split) > 1:
            mri = np.load(folder_path + x_split[0] + ".npy")
            if x_split[1][0:3] == "rev":
                mri = np.flipud(mri)
            elif x_split[1][0:3] == "rot": #rotation by degree
                #import pdb; pdb.set_trace()
                for frame in range(mri.shape[0]):
                    mri[frame,:,:] = rotateImage(mri[frame,:,:], float(x_split[1][3:]))
            if len(x_split) > 2: #rotate then reverse
                mri = np.flipud(mri)
        else: mri = np.load(folder_path + x + ".npy") # T x 256 x 256
        mri = uniform_seq_gen(mri, maxlen) # Pad sequences with 0s so they are all the same length per bucket
        mri = mri.reshape(mri.shape[0], mri.shape[1], mri.shape[2], 1)
        #print (x, mri.shape)
        return mri
    np_df = pd.concat([df, df["shahash"].apply(_load_matrix).to_frame(name="vectors")], axis=1)
    return np_df

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

"""
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
"""

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
    for m in videos:
        embeddings = conv_embedding_lookup(m, conv_wts, conv_bss, affine_wts, affine_bss, regularizer, is_training) # no embedding lookup --- maybe update to do end-to-end training :)
        embeddings_mri.append(embeddings)
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
        'accuracy': accuracy
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

    out = tf.nn.avg_pool(X, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #starts 256 x 256. ends 128 x128
    
    # conv relu batchnorm pool
    out = tf.nn.conv2d(out, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1 #starts 128 by 128 by 1, ends 124 by 124 by 32
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = is_training)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #62 by 62 by 32

    out = tf.nn.conv2d(out, Wconv2, strides=[1,1,1,1], padding='VALID') + bconv2 #starts 62 by 62 by 32, ends 58 by 58 by 32
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = is_training)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #29 by 29 by 32
    
    out = tf.nn.conv2d(out, Wconv3, strides=[1,1,1,1], padding='VALID') + bconv3 #starts 29 by 29, ends 25 by 25 by 32
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = is_training)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    
    out = tf.nn.conv2d(out, Wconv4, strides=[1,1,1,1], padding='VALID') + bconv4 #starts 25 by 25, ends 21 by 21 by 32
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = is_training)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "VALID") #10 by 10 by 32

    out = tf.reshape(out,[-1,800])
    
    out = tf.matmul(out,W_affine1024) + b_affine1024
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = is_training)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)

    out = tf.matmul(out,W_affine512) + b_affine512
    out = tf.nn.relu(out)
    #out = lrelu(out)
    out = tf.contrib.layers.batch_norm(out, is_training = is_training)
    out = tf.contrib.layers.dropout(out, p, is_training = is_training)
    
    return out

def train_graph(sess, saver, g, train_df, test_df, batch_size = 64, num_epochs = 10, iterator = BucketedDataIterator):
    sess.run(tf.global_variables_initializer())
    tr = iterator(train_df)
    te = iterator(test_df)
    step, accuracy = 0, 0
    tr_accs, te_accs, losses = [], [], []
    current_epoch = 0
    maxte_acc = 0
    validation_set_df = pd.DataFrame()
    while current_epoch < num_epochs:
        step += 1
        batch = tr.next_batch_imgs(batch_size)
        feed = {g['x']: batch[0], g['y']: list(batch[1]), g['seqlen']: list(batch[2]), g['is_training']: True}
        accuracy_, loss, scores, y, _ = sess.run([g['accuracy'], g['loss'], g['scores'], g['y'], g['ts']], feed_dict=feed)
        #print(batch[0].shape, batch[1].shape, batch[2].shape, loss)
        #break
        accuracy += accuracy_
        losses.append(loss)
        if tr.epochs > current_epoch:
            validation_set_df = pd.DataFrame()
            current_epoch += 1
            tr_accs.append(accuracy / step)
            step, accuracy = 0, 0
            #eval test set
            te_epoch = te.epochs
            while te.epochs == te_epoch:
                step += 1
                batch = te.next_batch_imgs(batch_size)
                #print ("test", batch[0].shape, batch[1].shape, batch[2].shape)
                feed = {g['x']: batch[0], g['y']: list(batch[1]), g['seqlen']: list(batch[2]), g['is_training']: False}
                accuracy_, preds, scores = sess.run([g['accuracy'], g['preds'], g['scores']], feed_dict=feed)
                #print (scores)
                bds = pd.concat([batch[1], batch[2], batch[3], pd.Series(preds).to_frame(name="preds")], axis=1)
                if validation_set_df.shape[0] == 0: validation_set_df = bds
                else: validation_set_df = pd.concat([validation_set_df, bds], axis=0)
                accuracy += accuracy_
            te_accs.append(accuracy / step)
            step, accuracy = 0,0
            print("Accuracy after epoch", current_epoch, " - tr:", tr_accs[-1], "- te:", te_accs[-1], " - loss", loss)
            sys.stdout.flush()
            if te_accs[-1] > maxte_acc:
                save_path = saver.save(sess, "CRNNbest.ckpt")
                maxte_acc = te_accs[-1]
            output_file.write("Accuracy after epoch: %f, current_epoch -tr: %f -te: %f\n" % (current_epoch, tr_accs[-1], te_accs[-1]))
    variables_names =[v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)

    fig = plt.figure()
    plt.plot(losses)
    fig.savefig(output_loss_file)
    plt.close(fig)
    validation_set_df.to_csv("validation_set_preds.tsv", sep="\t", index=None)
    return sess, tr_accs, te_accs, losses, variables_names, values

seq_identifer = "new_shape_reduced"
output_identifer = "output_val"
output_loss_file = "output_loss.png"
folder_path = "../../../only_tumor_frames/"
train_df = pd.read_csv("rnn_train_new_annotated.tsv", sep="\t")
test_df = pd.read_csv("val_new.tsv", sep="\t")
train_df = train_df[train_df["is_valid"] == 1]
test_df = test_df[test_df["is_valid"] == 1]
train_df = pd.concat([train_df, train_df["meth_stat"].apply(lambda x: 1 if x > 0 else 0).to_frame(name=output_identifer)], axis=1)
test_df = pd.concat([test_df, test_df["meth_stat"].apply(lambda x: 1 if x > 0 else 0).to_frame(name=output_identifer)], axis=1)

#for testing
#train_df = train_df[1:10]
#test_df = test_df[1:10]

train_copy = deepcopy(train_df)
test_copy = deepcopy(test_df)

r = global_config["rotations"] + 1
for angle in range(1, 89, 4):
    train_rev_df = deepcopy(train_copy)
    test_rev_df = deepcopy(test_copy)
    train_rev_df["shahash"] = train_rev_df["shahash"] + "_rot" + str(angle)
    test_rev_df["shahash"] = test_rev_df["shahash"] + "_rot" + str(angle)
    train_df = pd.concat([train_df, train_rev_df], axis=0)
    test_df = pd.concat([test_df, test_rev_df], axis=0)
    train_rev_df = deepcopy(train_copy)
    test_rev_df = deepcopy(test_copy)
    train_rev_df["shahash"] = train_rev_df["shahash"] + "_rot-" + str(angle)
    test_rev_df["shahash"] = test_rev_df["shahash"] + "_rot-" + str(angle)
    train_df = pd.concat([train_df, train_rev_df], axis=0)
    test_df = pd.concat([test_df, test_rev_df], axis=0)

#rev_everything 

train_rev_df = deepcopy(train_df)
test_rev_df = deepcopy(test_df)
train_rev_df["shahash"] = train_rev_df["shahash"] + "_rev"
test_rev_df["shahash"] = test_rev_df["shahash"] + "_rev"
train_df = pd.concat([train_df, train_rev_df], axis=0)
test_df = pd.concat([test_df, test_rev_df], axis=0)


print (train_df.shape)
print (test_df.shape)

output_file = open("output_file.txt", "w+")
batch_size = global_config["batch_size"]
g = build_conv_graph(batch_size=batch_size, state_size=global_config["rnn_size"]) #
saver = tf.train.Saver()
sess = tf.Session()
sess, tr_accs, te_accs, losses, variables_names, values = train_graph(sess, saver, g, train_df, test_df, batch_size=batch_size, num_epochs=global_config["epochs"], iterator=BucketedDataIterator)
output_file.close()

fig = plt.figure()
plt.plot(tr_accs, label="Training Acc.")
plt.plot(te_accs, label="Validation Acc.")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
fig.savefig("accuracies.png")
plt.close(fig)

fig = plt.figure()
plt.plot(losses, label="Softmax Cross-entropy Loss")
plt.xlabel("Iteration Number")
plt.ylabel("Loss")
plt.legend(loc='upper left')
fig.savefig("loss.png")
plt.close(fig)

save_path = saver.save(sess, "CRNN.ckpt")
#for k,v in zip(variables_names, values):
#    np.save(k + ".npy", v)
