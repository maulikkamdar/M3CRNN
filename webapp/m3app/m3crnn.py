from flask import Flask, Blueprint, render_template, json, request
import pandas as pd
import matplotlib.pyplot as plt, mpld3
import numpy as np
import tensorflow as tf
from utils import MatrixIO
import m3crnnmodels
import subprocess, time, hashlib

m3crnn = Blueprint('m3crnn', __name__, template_folder='templates')
app = Flask(__name__)
NUM_FILTERS = 8
user_dict = {}

def restore_model():
    g = m3crnnmodels.build_conv_graph(batch_size=1, state_size=m3crnnmodels.global_config["rnn_size"]) #
    saver = tf.train.Saver()
    sess = tf.Session()
    model_path = "./CRNNbest.ckpt"
    saver.restore(sess, model_path)
    print("Model restored.")
    return (saver, sess, g)

def run_scan(scan):
    iter_frame = np.floor(scan.shape[0]/16.0)
    feed = {g['x']: [scan.reshape(scan.shape[0], scan.shape[1], scan.shape[2], 1)], g['y']: [1], g['seqlen']: [scan.shape[0]], g['is_training']: False}
    firstnall, preds, scores = sess.run([g['firstnall'], g['preds'], g['scores']], feed_dict=feed)
    return firstnall, preds, scores

def which_data(data_type):
    if data_type == "train": df = training_data
    elif data_type == "valid": df = validation_data
    else: df = test_data
    return df

def visualize_layer_output(firstnall):
    '''@TODO deprecated'''
    html_snippets = {}
    for k in firstnall[0]:
        html_snippets[k] = {}
        for m in range(NUM_FILTERS):
            print str(m), k
            use_stack = firstnall[0][k][:,:,:,m]
            iter_frame = np.floor(use_stack.shape[0]/16.0)
            html_code = sample_stack(use_stack, rows=4, cols=4, show_every=int(iter_frame))
            html_snippets[k][m] = html_code 
    return html_snippets

def sample_stack(stack, rows=6, cols=6, start_with=0, show_every=1):
    fig, ax = plt.subplots(rows,cols,figsize=[12,12])
    #print fig
    for i in range(rows*cols):
        ind = start_with + i*show_every
        #print i, ind, stack.shape, show_every
        if i == rows*cols-1: ind = stack.shape[0]-1
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    #plt.show()
    html_code = mpld3.fig_to_html(fig, template_type="general")
    #print html_code
    return html_code

@m3crnn.route("/create_user")
def create_user():
    ip = request.args.get('ip')
    time_info = time.time()
    un_id = ip + ":-:" + str(time_info)
    user_id = "user_" + hashlib.sha1(un_id).hexdigest()
    user_dict[user_id] = {"time": time_info, "ip": ip}
    return json.dumps({"user_id": user_id})

@m3crnn.route("/m3crnn/make_decision")
def make_decision():
    shahash = request.args.get('shahash')
    data_type = request.args.get('q')
    is_valid = request.args.get('is_valid')
    seq_start = request.args.get("seq_start")
    seq_end = request.args.get("seq_end")
    tumor_start = request.args.get("tumor_start")
    tumor_end = request.args.get("tumor_end")
    print ("annotate", shahash, is_valid, seq_start, seq_end, tumor_start, tumor_end)
    f = open("output_file_decision_" + data_type + ".tsv", "a+")
    f.write(shahash + "\t" + data_type + "\t" + is_valid.title() + "\t" + seq_start + "\t" + seq_end + "\t" + tumor_start + "\t" + tumor_end + "\n")
    f.close()
    completed[shahash] = {"seq_end": seq_end, "seq_start": seq_start, "tumor_start": tumor_start, "tumor_end": tumor_end, "shahash": shahash, "data_type": data_type}
    completed[shahash]["is_valid"] = 1 if is_valid == 'true' else 0
    return json.dumps({"output": "Success"})

@m3crnn.route("/m3crnn/get_patient_scan")
def get_patient_scan():
    shahash = request.args.get('shahash')
    data_type = request.args.get('q')
    img_folder = data_folder + "tumorscans/"
    print (img_folder)
    #if ongoing_scan: subprocess.call(["rm", img_folder + ongoing_scan + ".npy"])
    #subprocess.call(["tar", "-zxvf", img_folder + "mriscans.tar.gz", img_folder + shahash + ".npy"])
    #ongoing_scan = shahash
    stack = np.load(img_folder + shahash + ".npy")
    iter_frame = np.ceil(stack.shape[0]/16.0)
    df = which_data(data_type)
    metadata = df[df["shahash"] == shahash]
    #print (stack.shape, df.shape, shahash)
    patient_id = list(metadata["Subject.ID"])[0]
    slice_count = stack.shape[0]
    meth_state = list(metadata["output_val"])[0]
    if shahash in completed:
        seq_end = completed[shahash]["seq_end"]
        seq_start = completed[shahash]["seq_start"]
        tumor_end = completed[shahash]["tumor_end"]
        tumor_start = completed[shahash]["tumor_start"]
    else:
        seq_end = slice_count
        tumor_end = slice_count
        seq_start = 0
        tumor_start = 0
    html_code = sample_stack(stack, rows=4, cols=4, start_with=0, show_every=int(iter_frame))
    return json.dumps({"html": html_code, "patient_id": patient_id, "slice_count": slice_count, "meth_state": meth_state, 
        "tumor_end": tumor_end, "tumor_start": tumor_start, "seq_end": seq_end, "seq_start": seq_start})

@m3crnn.route("/m3crnn/get_pred_scan")
def get_pred_scan():
    shahash = request.args.get('shahash')
    data_type = request.args.get('q')
    img_folder = data_folder + "testmris/"
    stack = np.load(img_folder + shahash + ".npy")
    print stack.shape
    iter_frame = np.floor(stack.shape[0]/16.0)
    metadata = ressha[ressha["id"] == shahash]
    print metadata
    patient_detail = list(metadata["desc"])[0]
    slice_count = stack.shape[0]
    print slice_count
    html_code = sample_stack(stack, rows=4, cols=4, start_with=0, show_every=int(iter_frame))
    return json.dumps({"html": html_code, "patient_id": patient_detail, "slice_count": slice_count})


@m3crnn.route("/m3crnn/save_annotations")
def save_annotations():
    mfio = MatrixIO()
    mfio.save_matrix(completed, data_folder + "completed.dat")
    return json.dumps({"output": "Success"})

@m3crnn.route("/m3crnn/get_patient_list")
def get_patient_list():
    data_type = request.args.get('q')
    def _prep_set(entries):
        data = []
        for k in entries:
            done = completed[k]["is_valid"] if k in completed else -1
            data.append({"shahash": k, "done": done})
        return data
    if data_type == "train": data = _prep_set(list(training_data["shahash"]))
    elif data_type == "valid": data = _prep_set(list(validation_data["shahash"]))
    else: data = _prep_set(list(test_data["shahash"]))
    return json.dumps({"data": data})

@m3crnn.route("/m3crnn/get_pred_list")
def get_pred_list():
    pred_type = request.args.get('q')
    def _prep_set(entries):
        data = []
        for k in entries:
            data.append({"shahash": entries[k]['id'], "label": entries[k]['desc']})
        return data
    data = ressha[ressha["label"] == pred_type.upper()]
    data = _prep_set(data[["id", "desc"]].to_dict(orient="index"))
    return json.dumps({"data": data})

@m3crnn.route("/m3crnn/visualize_output")
def visualize_filter_output():
    user_id = request.args.get('user_id')
    filter_info = request.args.get('filter_info')
    firstnall = user_dict[user_id]["firstnall"]
    fparts = filter_info.split("_")
    k = fparts[0].upper() + fparts[2].upper()
    m = int(fparts[1])
    print k, m
    use_stack = firstnall[0][k][:,:,:,m]
    iter_frame = np.floor(use_stack.shape[0]/16.0)
    html_code = sample_stack(use_stack, rows=4, cols=4, show_every=int(iter_frame))
    return json.dumps({"html": html_code})

@m3crnn.route("/m3crnn/visualize_mri")
def visualize_mri_scan():
    user_id = request.args.get('user_id')
    mri = user_dict[user_id]["mri"]
    iter_frame = np.floor(mri.shape[0]/16.0)
    html_code = sample_stack(mri, rows=4, cols=4, show_every=int(iter_frame))
    return json.dumps({"html": html_code})

@m3crnn.route("/m3crnn/run_scan")
def prepare_scan():
    shahash = request.args.get('shahash')
    data_type = request.args.get('q')
    user_id = request.args.get('user_id')
    if not user_id in user_dict: user_dict[user_id] = {}
    img_folder = data_folder + "tumorscans/"
    stack = np.load(img_folder + shahash + ".npy")
    df = which_data(data_type)
    metadata = df[df["shahash"] == shahash]
    patient_id = list(metadata["Subject.ID"])[0]
    slice_count = stack.shape[0]
    meth_state = list(metadata["output_val"])[0]
    print stack.shape
    if shahash in completed:
        tumor_end = completed[shahash]["tumor_end"]
        tumor_start = completed[shahash]["tumor_start"]
        stack = stack[int(tumor_start):int(tumor_end), :, :]
        firstnall, preds, scores = run_scan(stack)
        user_dict[user_id] = {"firstnall": firstnall, "preds": preds[0], "scores": scores[0], "mri": stack}
        score = round(float(list(scores[0])[int(preds[0])]), 6)
        return json.dumps({"output": "Success","patient_id": patient_id, "slice_count": slice_count, "meth_state": meth_state,  "scores": score, "preds": int(preds[0])})
    else:
        return json.dumps({"output": "Failure"})

data_folder = "static/data/m3crnn/"
training_data = pd.read_csv(data_folder + "trainset.tsv", sep="\t")
validation_data = pd.read_csv(data_folder + "validset_new.tsv", sep="\t")
test_data = pd.read_csv(data_folder + "testset_new.tsv", sep="\t")
ressha = pd.read_csv(data_folder + "ressha.tsv", sep="\t")
mfio = MatrixIO()
completed = mfio.load_matrix(data_folder + "completed.dat")
ongoing_scan = None
saver, sess, g = restore_model()

@m3crnn.route("/m3crnn/mri_viz")
def mri_viz():
    return render_template('m3crnn/mri_viz.html')

@m3crnn.route("/m3crnn/pred_viz")
def pred_viz():
    return render_template('m3crnn/pred_viz.html')

@m3crnn.route("/m3crnn/filter_viz")
def filter_viz():
    return render_template('m3crnn/filter_viz.html')

@m3crnn.route("/m3crnn/")
def index_main():
    return render_template('m3crnn/about.html')

@m3crnn.route("/m3crnn/about")
def m3crnnorig():
    return render_template('m3crnn/about.html')