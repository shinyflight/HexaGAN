import tensorflow as tf
import numpy as np

def drop_out(input, keep_prob, is_train):
    if is_train:
        out = tf.nn.dropout(input, keep_prob)
    else:
        keep_prob = 1
        out = tf.nn.dropout(input, keep_prob)

    return out

def _norm(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.layers.batch_normalization(inputs=input, training=is_train, reuse=reuse)
    else:
        out = input
    return out

def norm(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.layers.batch_normalization(inputs=input, training=is_train, reuse=reuse)
    else:
        out = input
    return out


def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.1)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    elif activation == 'prelu':
        alphas = tf.get_variable('alpha', input.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5
        return pos + neg
    else:
        return input

def pooling(input, k_size, stride, mode):
    assert mode in ['MAX', 'AVG']
    return tf.nn.max_pool(value=input,
                          ksize=[1, k_size[0], k_size[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                          padding='SAME',
                          name='max_pooling')

def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def conv2d(input, num_filters, filter_size, stride, reuse=False, pad='SAME', dtype=tf.float32, bias=True):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0,0],[p,p],[p,p],[0,0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)
        tf.nn.conv2d
    b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
    conv = conv + b
    return conv

def conv2d_transpose(input, num_filters, filter_size, stride, reuse, pad='SAME', dtype=tf.float32):
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, c]

    input_shape = tf.shape(input)
    try:  # tf pre-1.0 (top) vs 1.0 (bottom)
        output_shape = tf.pack([input_shape[0], stride * input_shape[1], stride * input_shape[2], num_filters])
    except Exception as e:
        output_shape = tf.stack([input_shape[0], stride * input_shape[1], stride * input_shape[2], num_filters])


    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)
    return deconv

def mlp(input, out_dim, name, is_train, reuse, norm=None, activation=None, dtype=tf.float32, bias=True):
    with tf.variable_scope(name, reuse=reuse):
        _, n = input.get_shape()
        w = tf.get_variable('w', [n, out_dim], dtype, tf.random_normal_initializer(0.0, 0.02))
        out = tf.matmul(input, w)

        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        out = out + b
        out = _activation(out, activation)
        out = _norm(out, is_train, reuse, norm)
        return out

def conv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out

def residual(input,  name, num_filters,  is_train, reuse, norm, activation, pad='SAME', bias=True):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2d(input, num_filters, 3, 1, reuse, pad, bias=bias)
            out = _norm(out, is_train, reuse, norm)
            out = _activation(out, activation)

        with tf.variable_scope('res2', reuse=reuse):
            out = conv2d(out, num_filters, 3, 1, reuse, pad, bias=bias)
            out = _norm(out, is_train, reuse, norm)
            out = _activation(out + input, activation)
        return out

def deconv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(input, num_filters, k_size, stride, reuse)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out


## Spectral normalization
def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def conv2d_sn(input, num_filters, filter_size, stride, reuse=False, pad='SAME', dtype=tf.float32, bias=True):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0,0],[p,p],[p,p],[0,0]], 'REFLECT')
        conv = tf.layers.conv2d(x, spectral_norm(w), stride_shape, padding='VALID')
        #conv = tf.layers.conv2d(x, num_filters, [filter_size, filter_size], [stride, stride], padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, spectral_norm(w), stride_shape, padding=pad)
        #conv = tf.layers.conv2d(input, num_filters, [filter_size, filter_size], [stride, stride], padding=pad)
    #b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
    #conv = conv + b
    return conv

def conv2d_transpose_sn(input, num_filters, filter_size, stride, reuse, pad='SAME', dtype=tf.float32):
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, c]

    n = tf.shape(input)[0]
    #input_shape = input.get_shape()
    output_shape = tf.stack([n, stride * h, stride * w, num_filters])

    weight = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, spectral_norm(weight), output_shape, stride_shape, pad)
    #deconv = tf.layers.conv2d_transpose(input, num_filters, [filter_size, filter_size], [stride, stride], pad)
    return deconv

def conv_block_sn(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_sn(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out

def deconv_block_sn(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose_sn(input, num_filters, k_size, stride, reuse)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out

def mlp_sn(input, out_dim, name, is_train, reuse, norm=None, activation=None, dtype=tf.float32, bias=True):
    with tf.variable_scope(name, reuse=reuse):
        _, n = input.get_shape()
        w = tf.get_variable('w', [n, out_dim], dtype, tf.random_normal_initializer(0.0, 0.02))
        out = tf.matmul(input, spectral_norm(w))

        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        out = out + b
        out = _activation(out, activation)
        out = _norm(out, is_train, reuse, norm)
        return out