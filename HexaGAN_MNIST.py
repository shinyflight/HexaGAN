'''
HexaGAN missing data imputation (MNIST)
Framework of this code is borrowed from Jinsung Yoon (https://github.com/jsyoon0823/GAIN/blob/master/MNST_Code_Example.py)
Weight clipping is used due to the high computational cost of zero centered gradient penalty
'''

# Packages
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ops import *
from tqdm import tqdm
from ops_cnn import *
tqdm.monitor_interval = 0

# GPU ID
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

save_dir = './result/HexaGAN_MNIST_missing_CNN/'
lr_GAN = 2e-3
decay = 1- (1e-5)
pretrain_steps = 0
d_steps = 1
weight_clip = True
# System Parameters
mb_size = 256
p_miss = 0.5
# Loss Hyperparameters
alpha = 10
Dim = 784
Train_No = 55000
Test_No = 10000

# Data Input
mnist = input_data.read_data_sets('./data', one_hot=True, seed=0)

# X
trainX, _ = mnist.train.next_batch(Train_No)
testX, _ = mnist.test.next_batch(Test_No)

# Mask Vector and Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C

trainM = sample_M(Train_No, Dim, p_miss)
testM = sample_M(Test_No, Dim, p_miss)

# Xavier Initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Plot
def plot(samples):
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(6, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

## Architecture
# Input Placeholders
X = tf.placeholder(tf.float32, shape=[None, Dim])
M = tf.placeholder(tf.float32, shape=[None, Dim])
Z = tf.placeholder(tf.float32, shape=[None, Dim])

# Generator (In this code, E and G_MI are integrated into one network)
def generator(X, z, m):
    DIM = 32
    with tf.variable_scope('generator'):
        x_tilde = m * X + (1 - m) * z  # Fill in random noise on the missing values
        x = tf.reshape(x_tilde, [-1, 28, 28, 1])
        m_ = tf.reshape(m, [-1, 28, 28, 1])
        input = tf.concat([x, m_], axis=3)
        output = conv_block(input, 'conv1', DIM, 5, 2, True, False, None, 'relu') # 14
        output = conv_block(output, 'conv2', 2 * DIM, 5, 2, True, False, None, 'relu') # 7
        output = conv_block(output, 'conv3', 4 * DIM, 5, 2, True, False, None, 'relu') # 4
        output = deconv_block(output, 'deconv1', 2 * DIM, 5, 2, True, False, None, 'relu')  # 8
        output = output[:, :7, :7, :] # 7
        output = deconv_block(output, 'deconv2', DIM, 5, 2, True, False, None, 'relu') # 14
        output = deconv_block(output, 'deconv3', 1, 5, 2, True, False, None, 'relu') # 28
        output = tf.reshape(output, [-1, Dim])
        output = linear(output, dim=Dim)
        x = tf.nn.sigmoid(output)
        G = m * X + (1 - m) * x  # Replace missing values to the imputed values
    return G, x

# Discriminator
def discriminator(X, reuse=True):
    DIM = 32
    with tf.variable_scope('discriminator', reuse=reuse):
        input = tf.reshape(X, [-1, 28, 28, 1])
        output = conv_block(input, 'conv1', DIM, 5, 2, True, False, None, 'relu') # 14
        output = conv_block(output, 'conv2', 2 * DIM, 5, 2, True, False, None, 'relu') # 7
        output = conv_block(output, 'conv3', 4 * DIM, 5, 2, True, False, None, 'relu')  # 4
        output = flatten(output)
        logits = mlp(output, Dim, 'fc4', True, False, None, None)
        D_prob = tf.nn.sigmoid(logits)
    return logits, D_prob

# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

# Structure
G_sample, G_sample_ = generator(X, Z, M)
D_logit, D_prob = discriminator(G_sample, reuse=False)

# Loss
D_loss1 = tf.reduce_mean(tf.div(tf.reduce_mean((1-M)*D_logit, 0),tf.reduce_mean(1-M,0))
                        -tf.div(tf.reduce_mean(M*D_logit, 0),tf.reduce_mean(M,0)))
G_loss1 = -tf.reduce_mean(tf.div(tf.reduce_mean((1-M)*D_logit, 0),tf.reduce_mean(1-M,0)))

MSE_train_loss = tf.reduce_mean((M * X - M * G_sample_) ** 2) / tf.reduce_mean(M)

variables = tf.trainable_variables()
theta_G = [var for var in variables if 'generator' in var.name]
theta_D = [var for var in variables if 'discriminator' in var.name]

# Weight clipping
clip_ops = []
for var in theta_D:
    clip_bounds = [-.01, .01]
    clip_ops.append(
        tf.assign(
            var,
            tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
        )
    )
D_clip_disc_weights = tf.group(*clip_ops)

D_loss = D_loss1
G_loss = G_loss1 + alpha * MSE_train_loss

# MSE Performance metric
MSE_test_loss = tf.reduce_mean(tf.sqrt(tf.div(tf.reduce_sum(((1-M)*X - (1-M)*G_sample) ** 2, 1), tf.reduce_sum(1-M, 1) + 1e-10)))

# Solver
global_step = tf.Variable(0, trainable=False)
lr_GAN = tf.train.exponential_decay(lr_GAN, global_step, 1, decay, staircase=True)

D_solver = tf.train.RMSPropOptimizer(lr_GAN).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.RMSPropOptimizer(lr_GAN).minimize(G_loss, var_list=theta_G, global_step=global_step)

print('model builded!')

# Sessions
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Output Initialization
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('train start!')
# Iteration Initialization
i = 1
best_rmse = 10000.0
# Start Iterations
for it in tqdm(range(10000)):

    # Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx, :]
    Z_mb = sample_Z(mb_size, Dim)
    M_mb = trainM[mb_idx, :]

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    if it < pretrain_steps:
        for d_step in range(d_steps):
            MSE_train_loss_curr, MSE_test_loss_curr, G_loss_curr, D_loss_curr, gp_curr = (100.0,)*5
        print('pretrain step:', it)
    else:
        for d_step in range(d_steps):
            _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={X: X_mb, M: M_mb, Z: New_X_mb})
            if weight_clip:
                _ = sess.run(D_clip_disc_weights)
                gp_curr=100.0
        _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run(
            [G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
            feed_dict={X: X_mb, M: M_mb, Z: New_X_mb})
        if best_rmse > MSE_test_loss_curr:
            best_rmse = MSE_test_loss_curr

    # Output figure
    if it % 100 == 0:
        figure_idx = np.array([3, 1, 4, 15, 49, 2, 20, 7, 6, 0])
        if it == 0:
            X_figure = testX[figure_idx, :]
            M_figure = testM[figure_idx, :]
            Z_figure = sample_Z(10, Dim)

        New_X_figure = M_figure * X_figure + (1 - M_figure) * Z_figure

        samples1 = X_figure
        samples5 = M_figure * X_figure + (1 - M_figure) * Z_figure

        samples2 = sess.run(G_sample, feed_dict={X: X_figure, M: M_figure, Z: New_X_figure})
        samples2 = M_figure * X_figure + (1 - M_figure) * samples2

        Z_figure = sample_Z(10, Dim)
        New_X_figure = M_figure * X_figure + (1 - M_figure) * Z_figure
        samples3 = sess.run(G_sample, feed_dict={X: X_figure, M: M_figure, Z: New_X_figure})
        samples3 = M_figure * X_figure + (1 - M_figure) * samples3

        Z_figure = sample_Z(10, Dim)
        New_X_figure = M_figure * X_figure + (1 - M_figure) * Z_figure
        samples4 = sess.run(G_sample, feed_dict={X: X_figure, M: M_figure, Z: New_X_figure})
        samples4 = M_figure * X_figure + (1 - M_figure) * samples4

        Z_figure = sample_Z(10, Dim)
        New_X_figure = M_figure * X_figure + (1 - M_figure) * Z_figure
        samples6 = sess.run(G_sample, feed_dict={X: X_figure, M: M_figure, Z: New_X_figure})

        samples = np.vstack([samples5, samples2, samples3, samples4, samples1, samples6])

        fig = plot(samples)
        plt.savefig(save_dir+'{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    # Intermediate Losses
    if it % 100 == 0:
        print('\nIter: {}'.format(it))
        print('Train_loss: {:.4};'.format(MSE_train_loss_curr))
        print('Test_loss: {:.4}; G_loss: {:.6}; D_loss: {:.6}'.format(MSE_test_loss_curr, G_loss_curr, D_loss_curr))
        print('Best RMSE: {:.4}'.format(best_rmse))
        print()