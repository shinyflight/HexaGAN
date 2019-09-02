import numpy as np
import tensorflow as tf
from copy import copy
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
import itertools


seed = np.random.randint(0, 100)
np.random.seed(seed)

## initializer
x_init = xavier_initializer()
v_init = variance_scaling_initializer()

## activation functions
def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

def relu_dropout(x, keep_prob):
    return tf.nn.dropout(tf.nn.relu(x), 1)

def lrelu(x , alpha = 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)


## layers
def batch_norm(input , is_training, scope):
    return tf.contrib.layers.batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, is_training=is_training, scope=scope, updates_collections=None)

def layer_norm(x, scope):
	with tf.variable_scope(scope):
		return slim.layer_norm(x, activation_fn=None)

def linear(h, dim, name='linear'):
    return tf.layers.dense(inputs=h, units=dim, kernel_initializer=x_init)

def imputation(z, X, m):
    return m * X + (1 - m) * z

def hint_mechanism(m, p):
    b   = np.random.choice([0,1], size=np.shape(m),   p=[1-p, p])
    h   = copy(m).astype(np.float32)
    h[np.where(b==1)] = 0.5
    return b, h


## loss
def WGAN_D_loss(D, M):
    real_D = D * M
    fake_D = D * (1-M)
    eps = 1e-8
    return tf.reduce_mean(tf.div(tf.reduce_mean(fake_D, 0),tf.reduce_mean(1-M,0)+eps)
                          -tf.div(tf.reduce_mean(real_D, 0),tf.reduce_mean(M,0)+eps))

def WGAN_G_loss(D, M):
    fake_D = D * (1-M)
    eps = 1e-8
    return -tf.reduce_mean(tf.div(tf.reduce_mean(fake_D, 0),tf.reduce_mean(1-M,0)+eps))

def softmax_CE(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))


## gradient penalty
def zc_gradient_penalty_D1(H, y, D1):
    _, pred = D1(H, y)
    gradients = tf.gradients(pred, H)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)) + 1e-8)
    gp = tf.reduce_mean((slopes) ** 2)
    return gp, slopes

def zc_gradient_penalty_D2(X, y, M, D2):
    d = X.shape[1].value
    _, D_logit = D2(X, y)
    gps = 0
    gradients = batch_jacobian(D_logit, X)
    slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), 2) + 1e-8)
    for i in range(d):
        print(i)
        gps += tf.reduce_sum(tf.square(slope[:,i]*M[:,i]))/(tf.reduce_sum(M[:,i])+1e-8)
    gp = gps/d
    return gp, slope


## mse
def RMSE(x, y, m, x_u, y_u, m_u):
    X = tf.concat([x, x_u], axis=0)
    Y = tf.concat([y, y_u], axis=0)
    M = tf.concat([m, m_u], axis=0)
    a = (1 - M) * X - (1 - M) * Y
    return tf.reduce_mean(tf.sqrt(tf.div(tf.reduce_sum(a ** 2, 1), tf.reduce_sum(1 - M, 1) + 1e-10)))

def MSE(x, y, m, x_u, y_u, m_u):
    X = tf.concat([x, x_u], axis=0)
    Y = tf.concat([y, y_u], axis=0)
    M = tf.concat([m, m_u], axis=0)
    a = M * X - M * Y
    return tf.reduce_mean(tf.div(tf.reduce_sum(a ** 2, 1), tf.reduce_sum(M, 1)+ 1e-10))


## miscellaneous
def clip(x):
    return tf.clip_by_value(x,1e-18,1.0)

def sample_z(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def concat(x, axis=1):
    return tf.concat(x, axis=axis)

def reverse(x):
    return (1-x)

def missing(x, m):
    return x*m

def onehot(y, y_dim):
    enc = OneHotEncoder(sparse=False)
    y_onehot = np.append(y,range(y_dim))
    y_onehot = y_onehot.reshape(len(y_onehot), 1)
    y_onehot = enc.fit_transform(y_onehot)
    return y_onehot[:-y_dim]

def calc_g(y, X_dim, y_dim):
    y_g = y
    m_g = np.zeros([len(y_g), X_dim+y_dim])
    return y_g, m_g

def balance(model, train_feed, X_dim, y_dim, z_dim, missing_p):
    train_feed_D2 = copy(train_feed)
    train_feed_C  = copy(train_feed)

    cnt_y = np.sum(train_feed[model.y], axis=0) * 1
    cnt_y = cnt_y.astype(int)
    y_g_list_ = [[i]*j for i, j in enumerate(cnt_y)]
    y_g_list = list(itertools.chain(*y_g_list_))
    y_g = onehot(y_g_list, y_dim)
    num_D2 = np.sum(cnt_y)

    ## D2
    train_feed_D2[model.y_g] = y_g
    train_feed_D2[model.z_G1] = sample_z(num_D2, z_dim)
    train_feed_D2[model.z_G2_g] = sample_z(num_D2, z_dim)
    train_feed_D2[model.m_g] = np.zeros([num_D2, X_dim+y_dim])

    ## C
    cnt_y = np.sum(train_feed[model.y], axis=0)
    y_C = np.max(cnt_y) - cnt_y
    y_C = y_C.astype(int)
    num_C = np.sum(y_C)
    y_C_list_ = [[i]*j for i, j in enumerate(y_C)]
    y_C_list = list(itertools.chain(*y_C_list_))
    train_feed_C[model.y_g] = onehot(y_C_list, y_dim)
    train_feed_C[model.z_G1] = sample_z(num_C, z_dim)
    train_feed_C[model.z_G2_g] = sample_z(num_C, z_dim)
    train_feed_C[model.m_g] = np.zeros([num_C, X_dim+y_dim])

    return train_feed_D2, train_feed_C

## data
def data_info(dataset, missing_p, batch_size, num_fold):
    # load data
    data_dir = './data/' + dataset + '_'
    mask = np.loadtxt(data_dir + 'mask_' + missing_p + '.csv', delimiter=',', dtype=bool)
    # calculate
    label_exist = mask[:,-1]
    num_X = np.count_nonzero(label_exist)
    num_X_u = len(label_exist) - num_X
    X_dim = np.shape(mask)[1]-1
    batch_size_u = int(batch_size * num_X_u / num_X * (num_fold-1)/num_fold)
    num_batches = int(int(num_X / batch_size)* (num_fold-1)/num_fold)
    return X_dim, batch_size_u, num_batches

def load_data(dataset, missing_p, model, sess, num_fold, max_epoch, batch_size, batch_size_u, num_batches, y_dim, z_dim, preproc='original'):
    # load data
    data_dir = './data/'+dataset+'_'
    if preproc == 'mice':
        Xy = np.loadtxt(data_dir + preproc + '_' + missing_p +'.csv', delimiter=',', dtype=np.float32)
    else:
        Xy = np.loadtxt(data_dir+preproc+'.csv', delimiter=',', dtype=np.float32)
    X_scaled = minmax_scale(Xy[:,:-1])

    mask = np.loadtxt(data_dir+'mask_'+missing_p+'.csv', delimiter=',', dtype=np.float32)
    mask_y = mask[:,-1].astype(bool) # to find labeled data
    mask_y_rev = reverse(mask_y).astype(bool) # to find unlabeled data
    X_data = X_scaled[mask_y]
    y_data = Xy[mask_y][:,-1]
    m_data = mask[mask_y]
    m_data = np.concatenate((m_data, np.repeat(np.expand_dims(m_data[:,-1], 1), y_dim - 1, axis=1)), axis=1)
    # shuffle
    np.random.seed(seed)
    rdn_shu = np.random.permutation(len(X_data))
    X_data = X_data[rdn_shu]
    y_data = y_data[rdn_shu]
    m_data = m_data[rdn_shu]

    X_u_data = X_scaled[mask_y_rev]
    m_u_data = mask[mask_y_rev]
    m_u_data = np.concatenate((m_u_data, np.repeat(np.expand_dims(m_u_data[:, -1], 1), y_dim - 1, axis=1)), axis=1)
    # shuffle
    np.random.seed(seed)
    rdn_shu = np.random.permutation(len(X_u_data))
    X_u_data = X_u_data[rdn_shu]
    m_u_data = m_u_data[rdn_shu]

    # one hot encoding
    y_data = onehot(y_data, y_dim)

    X_dim = np.shape(X_data)[1]

    k_fold = KFold(n_splits=num_fold)
    indices_l = k_fold.split(X_data)
    indices_u = k_fold.split(X_u_data)
    for i in range(num_fold):
        train_idx, test_idx = next(indices_l)
        train_X = X_data[train_idx]
        train_y = y_data[train_idx]
        train_m = m_data[train_idx]

        test_X_org = X_data[test_idx]
        test_y = y_data[test_idx]
        test_m = m_data[test_idx]
        test_X = missing(test_X_org, test_m[:,:-y_dim])

        train_idx_u, test_idx_u = next(indices_u)
        train_X_u = X_u_data[train_idx_u]
        train_m_u = m_u_data[train_idx_u]
        test_X_u_org = X_u_data[test_idx_u]
        test_m_u = m_u_data[test_idx_u]
        test_X_u = missing(test_X_u_org, test_m_u[:,:-y_dim])


        z_G2_test = sample_z(len(test_X), z_dim)
        z_G2_u_test = sample_z(len(test_X_u), z_dim)

        test_feed = {model.X: test_X, model.X_u: test_X_u, model.X_org: test_X_org, model.X_u_org: test_X_u_org,
                     model.y: test_y,
                     model.z_G2: z_G2_test, model.z_G2_u: z_G2_u_test,
                     model.m: test_m, model.m_u: test_m_u, model.training: False, model.keep_prob: 1.0}

        for j in range(max_epoch):
            # shuffle
            np.random.seed(j)
            rdn_shu_l = np.random.permutation(len(train_X))
            np.random.seed(j)
            rdn_shu_u = np.random.permutation(len(train_X_u))
            epoch_X = train_X[rdn_shu_l]
            epoch_y = train_y[rdn_shu_l]
            epoch_m = train_m[rdn_shu_l]
            epoch_X_u = train_X_u[rdn_shu_u]
            epoch_m_u = train_m_u[rdn_shu_u]

            for mini in range(num_batches):
                X_org_mb = epoch_X[mini*batch_size:(mini+1)*batch_size]
                y_mb = epoch_y[mini*batch_size:(mini+1)*batch_size]
                m_mb = epoch_m[mini*batch_size:(mini+1)*batch_size]

                X_mb = missing(X_org_mb, m_mb[:,:-y_dim])
                X_u_org_mb = epoch_X_u[mini*batch_size_u:(mini+1)*batch_size_u]
                m_u_mb = epoch_m_u[mini*batch_size_u:(mini+1)*batch_size_u]
                X_u_mb = missing(X_u_org_mb, m_u_mb[:,:-y_dim])

                # calculate y_g
                z_G2_u = sample_z(batch_size_u, z_dim)
                y_u_feed = {model.X_u: X_u_mb, model.z_G2_u:z_G2_u, model.m_u: m_u_mb, model.training: False, model.keep_prob: 1.0}
                y_u_mb = sess.run(model.pred_u, y_u_feed)
                y_g_mb, m_g_mb = calc_g(y_mb, X_dim, y_dim)

                # z
                z_G1   = sample_z(len(y_g_mb), z_dim)
                z_G2   = sample_z(len(y_mb),  z_dim)
                z_G2_g = sample_z(len(y_g_mb), z_dim)

                train_feed = {model.X: X_mb, model.X_u: X_u_mb, model.X_org: X_org_mb, model.X_u_org: X_u_org_mb,
                              model.y: y_mb, model.y_g: y_g_mb,
                              model.z_G1: z_G1, model.z_G2: z_G2, model.z_G2_u: z_G2_u, model.z_G2_g: z_G2_g,
                              model.m: m_mb, model.m_u: m_u_mb, model.m_g: m_g_mb,
                              model.training: True, model.keep_prob: 0.5}
                train_feed_D2, train_feed_C = balance(model, train_feed, X_dim, y_dim, z_dim, missing_p)

                yield train_feed, train_feed_D2, train_feed_C, test_feed