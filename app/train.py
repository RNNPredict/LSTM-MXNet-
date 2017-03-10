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
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


if __name__ == "__main__":

    # build char vocabluary from input
    vocab = build_vocab("../input.txt")
    print('vocab size = %d' %(len(vocab)))

    # Each line contains at most 129 chars. 
    seq_len = 129
    # embedding dimension, which maps a character to a 256-dimension vector
    num_embed = 256
    # number of lstm layers
    num_lstm_layer = 3
    # hidden unit in LSTM cell
    num_hidden = 512

    symbol = lstm.lstm_unroll(
        num_lstm_layer, 
        seq_len,
        len(vocab) + 1,
        num_hidden=num_hidden,
        num_embed=num_embed,
        num_label=len(vocab) + 1, 
        dropout=0.2)

    """test_seq_len"""
    data_file = open("../input.txt")
    for line in data_file:
        assert len(line) <= seq_len + 1, "seq_len is smaller than maximum line length. Current line length is %d. Line content is: %s" % (len(line), line)
    data_file.close()

    # The batch size for training
    batch_size = 32

    # initalize states for LSTM
    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    # Even though BucketSentenceIter supports various length examples,
    # we simply use the fixed length version here
    data_train = bucket_io.BucketSentenceIter(
        "../input.txt", 
        vocab,
        [seq_len], 
        batch_size,
        init_states, 
        seperate_char='\n',
        text2id=text2id,
        read_content=read_content)

    logging.getLogger().setLevel(logging.DEBUG)

    # We will show a quick demo with only 1 epoch. In practice, we can set it to be 100
    num_epoch = 1
    # learning rate 
    learning_rate = 0.01

    model = mx.model.FeedForward(
        ctx=mx.cpu(),
        symbol=symbol,
        num_epoch=num_epoch,
        learning_rate=learning_rate,
        momentum=0,
        wd=0.0001,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    model.fit(X=data_train,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 20),
              epoch_end_callback=mx.callback.do_checkpoint("model"))
