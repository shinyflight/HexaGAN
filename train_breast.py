import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn import metrics
import csv
import warnings
from model import *
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='breast')
parser.add_argument('--rate', type=str, default='0.2') # 0.0 to 0.9
args = parser.parse_args()

# set GPU ID
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# hyperparameters
save_bool = True # save results
save_bool_csv_name = args.dataset + '_' + args.rate
preproc = 'original'
dataset = args.dataset
missing_p = args.rate
log_every = 10
batch_size = 64
num_fold = 5

# acquire data information
X_dim, batch_size_u, num_batches = data_info(dataset, missing_p, batch_size, num_fold)

# learning rate
lr_C = 0.002
lr_GAN = 2e-4
decay = 1- (1e-5)

# epochs
pre_GD1_epoch = 0
pre_GD2_epoch = 60
max_epoch = 10000 + pre_GD1_epoch + pre_GD2_epoch

# update step
d_step = 1
g_step = 1

# dimension
y_dim = 2
H_dim = int(X_dim/2)
hidden_dim = int(X_dim/2)
z_dim = X_dim

# save file
if save_bool:
    save_path = './result/' + save_bool_csv_name + '.csv'
    if not(os.path.isdir('./result')):
            os.makedirs(os.path.join('./result'))
    if os.path.exists(save_path) == False:
        with open(save_path, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['epoch', 'test_rmse', 'test_pr', 'test_roc', 'test_f1', 'train_auc_g'])

# build model
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    model = HexaGAN(X_dim, z_dim, y_dim, H_dim, hidden_dim, lr_C, lr_GAN, decay, missing_p)
    model.build_model()
    saver = tf.train.Saver(max_to_keep=1000)

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
with tf.Session(graph=graph, config=config) as sess:
    init = tf.global_variables_initializer()
    init_loc = tf.local_variables_initializer()

    # declare generator for data loading
    batch = load_data(dataset, missing_p, model, sess, num_fold, max_epoch, batch_size, batch_size_u, num_batches, y_dim, z_dim, preproc)

    # training loop
    save_rmse_fold = []
    save_acc_fold = []
    save_pr_fold = []
    save_roc_fold = []
    save_f1_fold = []
    for i in range(num_fold):
        sess.run(init)
        sess.run(init_loc)
        print('Weights initialized\n')
        save_rmse_epoch = []
        save_acc_epoch = []
        save_pr_epoch = []
        save_roc_epoch = []
        save_f1_epoch = []
        best_f1 = 0
        for j in range(max_epoch):
            for mini in range(num_batches):
                E_loss=D1_loss=G1_loss=D2_loss=G2_loss=C_loss = 0
                # load batch
                train_feed, train_feed_D2, train_feed_C, test_feed = next(batch)

                if 0 <= j < pre_GD1_epoch:
                    for g in range(g_step):
                        _, E_loss = sess.run([model.E_opt, model.L_recon], train_feed)
                        _, G2_loss = sess.run([model.pre_G2_opt, model.L_recon], train_feed)
                        _ = sess.run([model.G1_opt], train_feed)
                    for d in range(d_step):
                        _, D1_loss = sess.run([model.D1_opt, model.D1_loss], train_feed)

                elif pre_GD1_epoch <= j < pre_GD1_epoch+pre_GD2_epoch:
                    for d in range(d_step):
                        _, D1_loss = sess.run([model.pre_D1_opt, model.pre_D1_loss], train_feed)
                        # update D2 network
                        _, D2_loss = sess.run([model.pre_D2_opt, model.pre_D2_loss], train_feed_D2)
                else:
                    for dn in range(10):
                        for g in range(g_step):
                            # update G1 network
                            _, G1_loss = sess.run([model.G1_opt, model.G1_loss], train_feed)
                        for d in range(d_step):
                            # update D1 network
                            _, D1_loss = sess.run([model.D1_opt, model.D1_loss], train_feed)

                    for g in range(g_step):
                        # update E network
                        _, E_loss  = sess.run([model.E_opt,  model.E_loss],  train_feed_D2)

                    for g in range(g_step):
                        # update G2 network
                        _, G2_loss = sess.run([model.G2_opt, model.G2_loss], train_feed_D2)

                    for d in range(d_step):
                        # update D2 network
                        _, D2_loss = sess.run([model.D2_opt, model.D2_loss], train_feed_D2)

                    # update C network
                    _, C_loss  = sess.run([model.C_opt,  model.C_loss],  train_feed_C)

            if ((j+1) % log_every == 0) & (j >= pre_GD1_epoch+pre_GD2_epoch):
                # calculate training performance
                train_acc, train_pred, train_rmse = sess.run([model.acc, model.C, model.rmse], train_feed)
                sess.run(init_loc)
                if y_dim < 3:
                    train_pr, _ = sess.run([model.pr_auc, model.pr_auc_op], train_feed)
                    train_roc = metrics.roc_auc_score(train_feed[model.y][:,1], train_pred[:,1])
                    train_f1 = metrics.f1_score(train_feed[model.y][:,1], np.argmax(train_pred, axis=1))
                if y_dim < 3:
                    print('fold: {}; epoch: {}; \nE_loss: {:.6f}; D1_loss: {:.6f}; G1_loss: {:.6f}; D2_loss: {:.6f}; G2_loss: {:.6f}; C_loss: {:.6f}; lr_GAN: {:.8f}\n'
                          'train_acc: {:.4f}; train_pr: {:.4f}; train_roc: {:.4f}; train_f1: {:.4f}; train_rmse: {};'
                          .format((i+1), (j+1), E_loss, D1_loss, G1_loss, D2_loss, G2_loss, C_loss, model.lr_GAN.eval(),
                          train_acc, train_pr, train_roc, train_f1, train_rmse))
                else:
                    print('fold: {}; epoch: {}; \nE_loss: {:.6f}; D1_loss: {:.6f}; G1_loss: {:.6f}; D2_loss: {:.6f}; G2_loss: {:.6f}; C_loss: {:.6f}; lr_GAN: {:.8f}\n'
                          'train_acc: {:.4f}; train_rmse: {};'
                          .format((i + 1), (j + 1), E_loss, D1_loss, G1_loss, D2_loss, G2_loss, C_loss,
                          model.lr_GAN.eval(), train_acc, train_rmse))
                # calculate test performance
                test_acc, test_pred, test_rmse = sess.run([model.acc, model.C, model.rmse], test_feed)
                sess.run(init_loc)
                if y_dim < 3:
                    test_pr, _ = sess.run([model.pr_auc, model.pr_auc_op], test_feed)
                    test_roc = metrics.roc_auc_score(test_feed[model.y][:,1], test_pred[:,1])
                    test_f1 = metrics.f1_score(test_feed[model.y][:,1], np.argmax(test_pred, axis=1))
                if y_dim < 3:
                    print('test_acc:  {:.4f}; test_pr: {:.4f};; test_roc: {:.4f}; test_f1: {:.4f}; test_rmse:  {};'
                          .format(test_acc, test_pr, test_roc, test_f1, test_rmse))
                else:
                    print('test_acc:  {:.4f}; test_rmse:  {};'.format(test_acc, test_rmse))
                save_rmse_epoch.append(test_rmse)
                save_acc_epoch.append(test_acc)
                if y_dim < 3:
                    save_pr_epoch.append(test_pr)
                    save_roc_epoch.append(test_roc)
                    save_f1_epoch.append(test_f1)
                    if best_f1 < test_f1:
                        best_f1 = test_f1
                    print('best_f1:', best_f1)
                print('best_rmse:', min(save_rmse_epoch))

                if y_dim < 3:
                    train_auc_g = metrics.roc_auc_score(train_feed[model.y_g], model.C_g.eval(train_feed))
                    print('train_auc_g',train_auc_g)
                    print('\n')
                    # save results
                    result = [j+1 - (pre_GD1_epoch+pre_GD2_epoch), test_rmse, test_pr, test_roc, test_f1, train_auc_g]
                    if save_bool:
                        with open(save_path, 'a') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',')
                            writer.writerow(result)
        save_rmse_fold.append(save_rmse_epoch)
        save_acc_fold.append(save_acc_epoch)
        if y_dim < 3:
            save_pr_fold.append(save_pr_epoch)
            save_roc_fold.append(save_roc_epoch)
            save_f1_fold.append(save_f1_epoch)


min_rmse = np.min(save_rmse_fold, axis=1)
min_rmse_idx = np.argmin(save_rmse_fold, axis=1)
max_acc = np.max(save_acc_fold, axis=1)
max_acc_idx = np.argmax(save_acc_fold, axis=1)
if y_dim < 3:
    max_pr = np.max(save_pr_fold, axis=1)
    max_pr_idx = np.argmax(save_pr_fold, axis=1)
    max_roc = np.max(save_roc_fold, axis=1)
    max_roc_idx = np.argmax(save_roc_fold, axis=1)
    max_f1 = np.max(save_f1_fold, axis=1)
    max_f1_idx = np.argmax(save_f1_fold, axis=1)
if save_bool:
    print('save_path:', save_path)
print('test_acc', max_acc)
print('max_acc_idx', (max_acc_idx+1)*log_every)
print('test_rmse', min_rmse)
print('min_rmse_idx', (min_rmse_idx+1)*log_every)
print('\nmean_test_rmse:  {:.4f}'.format(np.mean(min_rmse)))
print('mean_test_acc:  {:.4f}\n'.format(np.mean(max_acc)))
if y_dim < 3:
    print('test_pr', max_pr)
    print('max_pr_idx', (max_pr_idx+1)*log_every)
    print('test_roc', max_roc)
    print('max_roc_idx', (max_roc_idx+1)*log_every)
    print('test_f1', max_f1)
    print('max_f1_idx', (max_f1_idx+1)*log_every)
    print('\nmean_test_pr:  {:.4f}\n'.format(np.mean(max_pr)))
    print('mean_test_roc:  {:.4f}\n'.format(np.mean(max_roc)))
    print('mean_test_f1:  {:.4f}\n'.format(np.mean(max_f1)))
