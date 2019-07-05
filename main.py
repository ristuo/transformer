###############################################################################
# based on http://blog.varunajayasiri.com/ml/transformer.html
# the difference mainly is that instead of reading sequence length from
# tensor dimensions, it's passed a separate placeholder tensor around allowing
# sequence length to change from one minibatch to another. To that end some 
# variables like masks are also implemented in tf instead of turning numpy
# things to constant tensors.
###############################################################################
from random import shuffle
import math
import numpy as np
np.set_printoptions(linewidth=250)
import tensorflow as tf


def get_mean_std(x: tf.Tensor):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    squared = tf.square(x - mean)
    variance = tf.reduce_mean(squared, axis=-1, keepdims=True)
    std = tf.sqrt(variance)
    return mean, std


def layer_norm(layer: tf.Tensor, last_dim: int):
    with tf.variable_scope("norm" + layer.name.replace(":", "-")):
        scale = tf.get_variable(
            "scale", 
            shape=last_dim, 
            dtype=tf.float32
        )
        base = tf.get_variable("base", shape=last_dim, dtype=tf.float32)
        mean, std = get_mean_std(layer)
        norm = (layer - mean) / (std + 1e-6)
        return norm * scale + base


def attention(
    query: tf.Tensor, 
    key: tf.Tensor, 
    value: tf.Tensor, 
    *, 
    mask: tf.Tensor,
    keep_prob: float
):
    d_k = query.shape[-1].value
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) 
    scores = scores / tf.constant(math.sqrt(d_k))
    mask_add = ((scores * 0) - 1e9) * tf.constant(1.) - mask
    scores = scores * mask + mask_add
    attn = tf.nn.softmax(scores, axis=-1)
    attn = tf.nn.dropout(attn, keep_prob)
    return tf.matmul(attn, value), attn


def prepare_for_multi_head_attention(
    x: tf.Tensor, 
    heads: int, 
    name: str,
    n_batches: int,
    seq_len: tf.Tensor,
    d_model: int
):
    assert d_model % heads == 0
    d_k = d_model // heads
    x = tf.layers.dense(x, units=d_model, name=name)
    x = tf.reshape(x, shape=[n_batches, seq_len, heads, d_k])
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    return x


def multi_head_attention(
    query: tf.Tensor, 
    key: tf.Tensor, 
    value: tf.Tensor, 
    *,
    mask: tf.Tensor,
    heads: int,
    keep_prob: float,
    d_model: int,
    query_seq_len: tf.Tensor,
    kv_seq_len: tf.Tensor,
    n_batches: int
):
    with tf.variable_scope("multi_head"):
        query = prepare_for_multi_head_attention(
            x=query, 
            heads=heads, 
            name="query",
            d_model=d_model,
            seq_len=query_seq_len,
            n_batches=n_batches
        )
        key = prepare_for_multi_head_attention(
            key, 
            heads, 
            "key",
            d_model=d_model,
            seq_len=kv_seq_len,
            n_batches=n_batches
        )
        value = prepare_for_multi_head_attention(
            value, 
            heads, 
            "value",
            d_model=d_model,
            seq_len=kv_seq_len,
            n_batches=n_batches
        )
        mask = tf.expand_dims(mask, axis=1)
        out, _ = attention(query, key, value, mask=mask, keep_prob=keep_prob)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, shape=[n_batches, query_seq_len, d_model])
        return tf.layers.dense(out, units=d_model, name="attention")


def feed_forward(
    x: tf.Tensor,
    d_model: int, 
    d_ff: int, 
    keep_prob: float
):
    with tf.variable_scope("feed_forward"):
        hidden = tf.layers.dense(x, units=d_ff, name="hidden")
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        return tf.layers.dense(hidden, units=d_model, name="out")


def encoder_layer(x: tf.Tensor, 
                  *,
                  mask: tf.Tensor, 
                  index: int, 
                  heads: int,
                  keep_prob: float, 
                  d_ff: int,
                  d_model: int,
                  seq_len: tf.Tensor,
                  n_batches: int
):
    with tf.variable_scope("attention_{}".format(index)):
        attention_out = multi_head_attention(
            x, 
            x, 
            x,
            mask=mask, 
            heads=heads, 
            keep_prob=keep_prob,
            d_model=d_model,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            n_batches=n_batches
        )
        added = x + tf.nn.dropout(attention_out, keep_prob)
        x = layer_norm(added, d_model)
    with tf.variable_scope("ff_{}".format(index)):
        ff_out = feed_forward(x, d_model, d_ff, keep_prob)
        added = x + tf.nn.dropout(ff_out, keep_prob)
        return layer_norm(added, d_model)


def encoder(
    x: tf.Tensor, *,
    mask: tf.Tensor,
    n_layers: int,
    heads: int, 
    keep_prob: float, 
    d_ff: int,
    d_model: int,
    n_batches: int,
    seq_len: tf.Tensor
):
    with tf.variable_scope("encoder"):
        for i in range(n_layers):
            x = encoder_layer(
                x,
                mask=mask, 
                index=i,
                heads=heads, 
                keep_prob=keep_prob, 
                d_ff=d_ff,
                d_model=d_model,
                seq_len=seq_len,
                n_batches=n_batches
            )
        return x


def decoder_layer(
    encoding: tf.Tensor, 
    x: tf.Tensor, 
    *,
    enc_mask: tf.Tensor, 
    mask: tf.Tensor,
    index: int, 
    heads: int, 
    keep_prob: float, 
    d_ff: int,
    d_model: int,
    x_seq_len: tf.Tensor,
    y_seq_len: tf.Tensor,
    n_batches: int
):
    with tf.variable_scope("{}_self_attention".format(index)):
        attention_out = multi_head_attention(
            x, 
            x, 
            x,
            mask=mask, 
            heads=heads, 
            keep_prob=keep_prob,
            d_model=d_model,
            query_seq_len=y_seq_len,
            kv_seq_len=y_seq_len,
            n_batches=n_batches
        )
        added = x + tf.nn.dropout(attention_out, keep_prob=keep_prob)
        x = layer_norm(added, d_model)
    
    with tf.variable_scope("{}_encoding_attention".format(index)):
        attention_out = multi_head_attention(
            x, 
            encoding, 
            encoding,
            mask=enc_mask, 
            heads=heads, 
            keep_prob=keep_prob,
            d_model=d_model,
            query_seq_len=y_seq_len,
            kv_seq_len=x_seq_len,
            n_batches=n_batches
        )
        added = x + tf.nn.dropout(attention_out, keep_prob=keep_prob)
        x = layer_norm(added, d_model)

    with tf.variable_scope("{}_ff".format(index)):
        ff_out = feed_forward(x, d_model, d_ff, keep_prob)
        added = x + tf.nn.dropout(ff_out, keep_prob)
        return layer_norm(added, d_model)


def decoder(
    encoding: tf.Tensor, 
    x: tf.Tensor, 
    *,
    enc_mask: tf.Tensor, 
    mask: tf.Tensor,
    n_layers: int,
    heads: int, 
    keep_prob: float, 
    d_ff: int,
    d_model: int,
    x_seq_len: tf.Tensor,
    y_seq_len: tf.Tensor,
    n_batches: int
):
    with tf.variable_scope("decoder"):
        for i in range(n_layers):
            x = decoder_layer(
                encoding, 
                x,
                enc_mask=enc_mask, 
                mask=mask, 
                index=i,
                heads=heads, 
                keep_prob=keep_prob, 
                d_ff=d_ff,
                d_model=d_model,
                x_seq_len=x_seq_len,
                y_seq_len=y_seq_len,
                n_batches=n_batches
            )
        return x


def get_embeddings(
    input_ids: tf.Tensor, 
    output_ids: tf.Tensor,
    vocab_size: int, 
    d_model: int,
    max_input_seq_len: tf.Tensor,
    max_y_seq_len: tf.Tensor,
    n_batches: int
):
    word_embeddings = tf.get_variable(
        "word_embeddings",
        shape=[vocab_size, d_model],
        dtype=tf.float32,
        initializer=tf.initializers.random_normal()
    )
    in_emb = tf.nn.embedding_lookup(word_embeddings, input_ids)
    in_emb = tf.reshape(in_emb, [n_batches, max_input_seq_len, d_model])
    out_emb = tf.nn.embedding_lookup(word_embeddings, output_ids)
    out_emb = tf.reshape(out_emb, [n_batches, max_y_seq_len, d_model])
    return word_embeddings, in_emb, out_emb



def generate_positional_encodings(d_model: int, max_len: int = 5000):
    encodings = np.zeros((max_len, d_model), dtype=float)
    position = np.arange(0, max_len).reshape((max_len, 1))
    two_i = np.arange(0, d_model, 2)
    div_term = np.exp(-math.log(10000.0) * two_i / d_model)
    encodings[:, 0::2] = np.sin(position * div_term)
    encodings[:, 1::2] = np.cos(position * div_term)
    return tf.constant(
        encodings.reshape((1, max_len, d_model)),
        dtype=tf.float32, 
        name="positional_encodings"
    )

def prepare_embeddings(
    x: tf.Tensor, 
    *,
    positional_encodings: tf.Tensor,
    keep_prob: float, 
    is_input: bool,
    seq_len: tf.Tensor,
    d_model: int
):
    name = "prepare_input" if is_input else "prepare_output"
    with tf.variable_scope(name):
        x = x + positional_encodings[:, :seq_len, :]
        x = tf.nn.dropout(x, rate = 1 - keep_prob)
        return layer_norm(x, d_model)


def generator(x: tf.Tensor, *, vocab_size: int):
    res = tf.layers.dense(x, units=vocab_size, name="generator")
    return tf.nn.log_softmax(res, axis=-1)


def label_smoothing_loss(
    results: tf.Tensor, 
    expected: tf.Tensor, 
    *,
    vocab_size: int, 
    smoothing: float
):
    results = tf.reshape(results, shape=(-1, vocab_size))
    expected = tf.reshape(expected, shape=[-1])
    confidence = 1 - smoothing
    smoothing = smoothing / (vocab_size - 1)
    expected = tf.one_hot(expected, depth=vocab_size) * (confidence - smoothing)
    expected += smoothing
    results = tf.distributions.Categorical(logits=results)
    expected = tf.distributions.Categorical(logits=expected)
    return tf.reduce_mean(tf.distributions.kl_divergence(results, expected))


def noam_learning_rate(step: int, warm_up: float, d_model: int):
    return (d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)


def output_subsequent_mask(max_input_seq_len: tf.Tensor, name):
    matrix_of_ones = tf.ones(
        (1, max_input_seq_len, max_input_seq_len), 
        name=name
    )
    return tf.linalg.band_part(matrix_of_ones, -1,0)


def product(xs):
    res = 1
    for x in xs:
        res *= x
    return res
variables = tf.global_variables()
parameters = sum([product(v.shape) for v in variables]).value

def __print_seq(seq, index_to_char):
    return ' '.join([index_to_char[i] for i in seq])



def to_numeric(xs, char_to_index):
    return [char_to_index[x] for x  in xs]


def to_text(xs, index_to_char):
    return [index_to_char[x] for x in xs]


def make_records(texts, char_to_index, context_size=15):
    res = []
    for i in range(context_size, len(texts)):
        x = to_numeric("\n".join(texts[(i - context_size): i]), char_to_index)
        y = to_numeric(texts[i], char_to_index)
        res += [(x,y)]
    return res
    

def load_data():
    with open("data/kalevala.txt", "rb") as fp:
        text = fp.read().decode()
    vocabulary = list(set(text)) + ["_", "$"]
    char_to_index = dict(zip(vocabulary, range(len(vocabulary))))
    index_to_char = {v:k for k,v in char_to_index.items()}
    text = text.replace("\r","").split("\n")
    text = [x.strip() for x in text if x != ""]
    n_records = len(text)
    cutoff = int(n_records * (1 - 0.05))
    training = make_records(text[:cutoff], char_to_index)
    test = make_records(text[cutoff:], char_to_index, context_size=5)
    return training, test, char_to_index, index_to_char, char_to_index["_"], char_to_index["$"]


def get_batch(training, i, batch_size, pad_number, start_number):
    def pad_left(x, to_size, pad_number):
        return [pad_number] * (to_size - len(x)) + x
    def pad_right(x, to_size, pad_number):
        return x + [pad_number] * (to_size - len(x)) 
    j = (i * batch_size) % len(training)
    end = j + batch_size
    batch = training[j:end]
    x_lengths = [len(x) for x,_ in batch]
    y_lengths = [len(y) for _,y in batch]
    batch_max_len_x = max(x_lengths)
    batch_max_len_y = max(y_lengths)
    padded = [
        (
            pad_left(x, batch_max_len_x, pad_number), 
            pad_right(y, batch_max_len_y, pad_number)
        )
        for x,y in batch
    ]
    return [(x, [start_number] + y) for x,y in padded]



warm_up = 400
d_model = 128  # 512
heads = 8
keep_prob = 0.9
n_layers = 3  # 6
d_ff = 256  # 2048
batch_size = 16

positional_encodings = generate_positional_encodings(d_model)

inputs = tf.placeholder(
    dtype=tf.int32,
    name="input"
)
max_input_seq_len = tf.placeholder(
    dtype=tf.int32, 
    name="max_input_seq_len"
)
max_y_seq_len = tf.placeholder(
    dtype=tf.int32, 
    name="max_y_seq_len"
)
outputs = tf.placeholder(
    dtype=tf.int32,
    name="output"
)
expected = tf.placeholder(
    dtype=tf.int32,
    name="expected"
)
inputs_mask = tf.ones(
    (1, 1, max_input_seq_len), 
    dtype=tf.float32, 
    name="input_mask"
)
output_mask = output_subsequent_mask(max_y_seq_len, "output_mask") 
learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
training, test, char_to_index, index_to_char, pad_number, start_number = load_data()
shuffle(training)
vocab_size = len(index_to_char)

_, input_embeddings, output_embeddings = get_embeddings(
    inputs, 
    outputs, 
    vocab_size,
    d_model=d_model,
    max_input_seq_len=max_input_seq_len,
    max_y_seq_len=max_y_seq_len,
    n_batches=batch_size
)
prepared_input_embeddings = prepare_embeddings(
    input_embeddings,
    positional_encodings=positional_encodings,
    keep_prob=keep_prob,
    is_input=True,
    seq_len=max_input_seq_len,
    d_model=d_model
)
prepared_output_embeddings = prepare_embeddings(
    output_embeddings,
    positional_encodings=positional_encodings,
    keep_prob=keep_prob,    
    is_input=False,
    seq_len=max_y_seq_len,
    d_model=d_model
)
encoding = encoder(
    prepared_input_embeddings, 
    mask=inputs_mask, 
    n_layers=n_layers, 
    heads=heads,
    keep_prob=keep_prob, 
    d_ff=d_ff,
    d_model=d_model,
    seq_len=max_input_seq_len,
    n_batches=batch_size
)
decoding = decoder(
    encoding, 
    prepared_output_embeddings,
    enc_mask=inputs_mask, 
    mask=output_mask,
    n_layers=n_layers, 
    heads=heads, 
    keep_prob=keep_prob, 
    d_ff=d_ff,
    d_model=d_model,
    y_seq_len=max_y_seq_len,
    x_seq_len=max_input_seq_len,
    n_batches=batch_size
)

log_results = generator(decoding, vocab_size=vocab_size)
results = tf.exp(log_results)

loss = label_smoothing_loss(
    log_results, 
    expected, 
    vocab_size=vocab_size, 
    smoothing=0.0
)
adam = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
params = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.)
grads_and_vars = list(zip(grads, params))
train_op = adam.apply_gradients(grads_and_vars, name="apply_gradients")

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        lr = noam_learning_rate(i + 1, warm_up, d_model)
        batch = get_batch(training, i, batch_size, pad_number, start_number)
        if len(batch) != batch_size:
            continue
        batch_seq_len = len(batch[0][0])
        batch_y_seq_len = len(batch[0][1]) - 1
        batch_in = np.array([b[0] for b in batch])
        batch_out = np.array([b[1] for b in batch])
        feed_outputs = batch_out[:, :-1]
        feed_inputs = batch_in
        feed_expected = batch_out[:, 1:]
        _, batch_loss, batch_res = sess.run(
            [train_op, loss, results],
            feed_dict={
                learning_rate: lr,
                inputs: feed_inputs,
                outputs: feed_outputs,
                expected: feed_expected,
                max_input_seq_len: batch_seq_len,
                max_y_seq_len: batch_y_seq_len
            }
        )

        if i % 50 == 0:
            test_batch = get_batch(test, 0, batch_size, pad_number, start_number)
            test_batch_in = np.array([b[0] for b in test_batch])
            test_batch_out = np.array([b[1] for b in test_batch])
            test_feed_outputs = test_batch_out[:, :-1]
            test_feed_inputs = test_batch_in
            test_example = sess.run(
                results,
                feed_dict={
                    learning_rate: lr,
                    inputs: test_feed_inputs,
                    outputs: test_feed_outputs,
                    max_input_seq_len: len(test_batch[0][0]),
                    max_y_seq_len: len(test_batch[0][1]) - 1 
                }
            )
            print("step={}: loss={}".format(i, batch_loss))
            print("inp={}".format(__print_seq(feed_inputs[0], index_to_char)))
            print("out={}".format(__print_seq(feed_outputs[0], index_to_char)))
            print("exp={}".format(__print_seq(feed_expected[0], index_to_char)))
            print("res={}".format(__print_seq(np.argmax(batch_res[0], -1), index_to_char)))
            print("tst={}".format(__print_seq(test_feed_outputs[0], index_to_char)))
            print("tre={}".format(__print_seq(np.argmax(test_example[0], -1), index_to_char)))
# todo: 
#   - tensorboard monitoring
#   - computing actual validation loss          
