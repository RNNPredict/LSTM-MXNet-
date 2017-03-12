import os
import bucket_io
import lstm
import mxnet as mx
import numpy as np
import logging

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

# Encode a sentence with int ids
def text2id(sentence, the_vocab):
    words = list(sentence)
    return [the_vocab[w] for w in words if len(w) > 0]

# Evaluation metric
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

def train(seq_len, num_embed, num_lstm_layer, num_hidden, batch_size,
          num_epoch, learning_rate, gpus, vocab_file, param_file):
    """Training."""

    # build char vocabluary from input
    vocab = build_vocab(vocab_file)
    logging.info('vocab size = %d', len(vocab))

    symbol = lstm.lstm_unroll(
        num_lstm_layer, 
        seq_len,
        len(vocab) + 1,
        num_hidden=num_hidden,
        num_embed=num_embed,
        num_label=len(vocab) + 1, 
        dropout=0.2)

    """test_seq_len"""
    data_file = open(vocab_file)
    for line in data_file:
        assert len(line) <= seq_len + 1, "seq_len is smaller than maximum line length. Current line length is %d. Line content is: %s" % (len(line), line)
    data_file.close()

    # initalize states for LSTM
    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    # Even though BucketSentenceIter supports various length examples,
    # we simply use the fixed length version here
    data_train = bucket_io.BucketSentenceIter(
        vocab_file,
        vocab,
        [seq_len], 
        batch_size,
        init_states, 
        seperate_char='\n',
        text2id=text2id,
        read_content=read_content)

    # context
    ctx = mx.cpu()
    if len(gpus) > 0:
        ctx = [mx.gpu(int(i)) for i in gpus.split(',')]

    # buckets = [10, 20, 30, 40, 50, 60]
    # buckets = [32]
    # state_names = [x[0] for x in init_states]
    # def sym_gen(seq_len):
    #     data_names = ['data'] + state_names
    #     label_names = ['softmax_label']
    #     return (symbol, data_names, label_names)
    #
    # if len(buckets) == 1:
    #     mod = mx.mod.Module(*sym_gen(buckets[0]), context=ctx)
    # 
    #     head = '%(asctime)-15s %(message)s'
    #     logging.basicConfig(level=logging.DEBUG, format=head)
    # 
    #     mod.fit(data_train, num_epoch=num_epoch, eval_data=data_train,
    #         eval_metric=mx.metric.np(Perplexity),
    #         batch_end_callback=mx.callback.Speedometer(batch_size, 50),
    #         epoch_end_callback=mx.callback.do_checkpoint(param_file),
    #         initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
    #         optimizer='sgd',
    #         optimizer_params={'learning_rate': learning_rate, 'momentum': 0.9, 'wd': 0.00001})
    # 
    #     Now it is very easy to use the bucketing to do scoring or collect prediction outputs
    #     metric = mx.metric.np(Perplexity)
    #     mod.score(data_val, metric)
    #     for name, val in metric.get_name_value():
    #         logging.info('Validation-%s=%f', name, val)
    # 
    # else:

    model = mx.model.FeedForward(
        ctx=ctx,
        symbol=symbol,
        num_epoch=num_epoch,
        learning_rate=learning_rate,
        momentum=0,
        wd=0.0001,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    model.fit(X=data_train,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 20),
              epoch_end_callback=mx.callback.do_checkpoint(param_file))


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)

    # Each line contains at most xxx chars. 
    seq_len = int(os.getenv("SEQUENCE_LEN", "130"))
    logging.info('sequence length = %d', seq_len)
    # embedding dimension, which maps a character to a 256-dimension vector
    num_embed = 256
    # number of lstm layers
    num_lstm_layer = int(os.getenv("LSTM_LAYERS", "3"))
    logging.info('lstm layers = %d', num_lstm_layer)
    # hidden unit in LSTM cell
    num_hidden = int(os.getenv("UNITS_IN_CELL", "512"))
    logging.info('unit in LSTM cell = %d', num_hidden)
    # The batch size for training
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    logging.info('batch size = %d', batch_size)
    # In practice, we can set it to be 100
    num_epoch = int(os.getenv("LEARNING_EPOCHS", "10"))
    logging.info('epoch = %d', num_epoch)
    # learning rate
    learning_rate = float(os.getenv("LEARNING_RATE", "0.01"))
    logging.info('learning rate = %f', learning_rate)
    # context
    gpus = os.getenv("GPUS", "")
    logging.info('GPUs = %s', gpus)

    # vocabulary file
    vocab_file = os.getenv("VOCABULARY_FILE", "./input.txt")
    logging.info('vocabulary file = %s', vocab_file)
    # model
    param_file = os.getenv("PARAMETERS_FILE", "model")
    logging.info('parameters file = %s', param_file)

    train(seq_len=seq_len, num_embed=num_embed, num_lstm_layer=num_lstm_layer,
          num_hidden=num_hidden, batch_size=batch_size, num_epoch=num_epoch,
          learning_rate=learning_rate, gpus=gpus, vocab_file=vocab_file,
          param_file=param_file)
