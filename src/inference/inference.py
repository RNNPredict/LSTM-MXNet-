import os
import mxnet as mx
import numpy as np
import rnn_model
import random
import bisect
import logging

from flask import Flask, g, abort, redirect
app = Flask(__name__)

def read_content(path):
    with open(path) as ins:
        return ins.read()

# Return a dict which maps each char into an unique int id
def build_vocab(path):
    content = list(read_content(path))
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab

# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic

# make input from char
def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp

# helper function for random sample 
def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if sample == False:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
        scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
        rescale = np.exp(np.log(scale_prob) / temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char = ''
    return char

def inferenceModel(vocab, num_embed, num_lstm_layer, num_hidden,
                   num_epoch, gpus, vocab_file, param_file):
    """Inference."""

    # load from check-point
    _, arg_params, __ = mx.model.load_checkpoint(param_file, num_epoch)

    # context
    ctx = mx.cpu()
    if len(gpus) > 0:
        ctx = [mx.gpu(int(i)) for i in gpus.split(',')]

    # build an inference model
    return rnn_model.LSTMInferenceModel(
        num_lstm_layer,
        len(vocab) + 1,
        num_hidden=num_hidden,
        num_embed=num_embed,
        num_label=len(vocab) + 1, 
        arg_params=arg_params, 
        ctx=ctx, 
        dropout=0.2)

def inference(_model, vocab, start_with, seq_len):
    input_ndarray = mx.nd.zeros((1,))
    revert_vocab = MakeRevertVocab(vocab)

    output = start_with
    random_sample = True
    new_sentence = True

    ignore_length = len(output)

    for i in range(seq_len):
        if i <= ignore_length - 1:
            MakeInput(output[i], vocab, input_ndarray)
        else:
            MakeInput(output[-1], vocab, input_ndarray)
        prob = _model.forward(input_ndarray, new_sentence)
        new_sentence = False
        next_char = MakeOutput(prob, revert_vocab, random_sample)
        if next_char == '':
            new_sentence = True
        if i >= ignore_length - 1:
            output += next_char
    return output

@app.route('/')
def ok():
    return redirect('/AWS%20is')

@app.route('/favicon.ico')
def favicon():
    abort(404)

cache = {}

# inference
@app.route('/<start>')
def get(start=None):
    return inference(cache['model'], cache['vocab'], start, cache['seq_len'])

@app.before_first_request
def app_init():

    # The output is expected to start with...
    start_with = os.getenv("START_WITH", "Amazon EC2 is")
    logging.info('start with = %s' % start_with)
    # Each line contains at most xxx chars. 
    seq_len = int(os.getenv("SEQUENCE_LEN", "300"))
    cache['seq_len'] = seq_len
    logging.info('sequence length = %d' % seq_len)
    # embedding dimension, which maps a character to a 256-dimension vector
    num_embed = 256
    # number of lstm layers
    num_lstm_layer = int(os.getenv("LSTM_LAYERS", "3"))
    logging.info('lstm layers = %d' % num_lstm_layer)
    # hidden unit in LSTM cell
    num_hidden = int(os.getenv("UNITS_IN_CELL", "512"))
    logging.info('unit in LSTM cell = %d' % num_hidden)
    # In practice, we can set it to be 100
    num_epoch = int(os.getenv("LEARNING_EPOCHS", "10"))
    logging.info('epoch = %d' % num_epoch)
    # context
    gpus = os.getenv("GPUS", "")
    logging.info('GPUs = %s' % gpus)

    # vocabulary file
    vocab_file = os.getenv("VOCABULARY_FILE", "./input.txt")
    logging.info('vocabulary file = %s' % vocab_file)
    # model
    param_file = os.getenv("PARAMETERS_FILE", "model")
    logging.info('parameters file = %s' % param_file)

    # build char vocabluary from input
    vocab = build_vocab(vocab_file)
    cache['vocab'] = vocab
    logging.info('vocab size = %d' %(len(vocab)))

    # make the model
    model = inferenceModel(vocab, num_embed, num_lstm_layer, num_hidden,
                           num_epoch, gpus, vocab_file, param_file)
    cache['model'] = model


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)
    app.run(host='0.0.0.0')
