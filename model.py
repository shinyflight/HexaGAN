from ops import *

class HexaGAN(object):
    def __init__(self, X_dim, z_dim, y_dim, H_dim, hidden_dim, lr_C, lr_GAN, decay, missing_p):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.H_dim = H_dim
        self.lr_C = lr_C
        self.lr_GAN = lr_GAN
        self.decay = decay
        self.missing_p = float(missing_p)

    ## HexaGAN components
    # Generator for conditional generation (G_CG)
    def Generator1(self, y, z, reuse=True, scope='generator1'):
        with tf.variable_scope(scope, reuse=reuse):
            x = concat([y, z])
            x = tf.nn.relu(linear(x, dim=self.hidden_dim))
            H = tf.nn.relu(linear(x, dim=self.H_dim))
        return H

    # Encoder
    def Encoder(self, X, m, z, reuse=True, scope='encoder'):
        with tf.variable_scope(scope, reuse=reuse):
            x = m * X + (1-m) * z
            x = tf.concat([x, m], axis=1)
            x = tf.nn.relu(linear(x, dim=self.hidden_dim))
            H = tf.nn.relu(linear(x, dim=self.H_dim))
        return H

    # Discriminator for conditional generation (D_CG)
    def Discriminator1(self, h, y, reuse=True, scope='discriminator1'):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.concat([h, y], 1)
            x = tf.nn.relu(linear(x, dim=self.hidden_dim))
            logits = linear(x, dim=1)
            D = tf.nn.sigmoid(logits)
        return D, logits

    # Generator for missing imputation (G_MI)
    def Generator2(self, X, m, H, reuse=True, scope='generator2'):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.nn.relu(linear(H, dim=self.hidden_dim))
            x = sigmoid(linear(x, dim=self.X_dim))
            X_= m*X + (1-m)*x
        return X_, x

    # Discriminator for missing imputation (D_MI)
    def Discriminator2(self, X, y, reuse=True, scope='discriminator2'):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.concat([X, y], 1)
            x = tf.nn.relu(linear(x, dim=self.hidden_dim))
            logits = linear(x, dim=self.X_dim+self.y_dim)
            D = tf.nn.sigmoid(logits)
        return D, logits

    # Classifier (also acts as label generator)
    def Classifier(self, X_, reuse=True, scope='classifier'):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.nn.relu(linear(X_, dim=self.hidden_dim))
            logits = linear(x, dim=self.y_dim)
            pred = softmax(logits)
        return pred, logits


    def build_model(self):
        ## Placeholders
        # X
        self.X   = tf.placeholder(tf.float32, [None, self.X_dim])
        self.X_u = tf.placeholder(tf.float32, [None, self.X_dim])
        # for mse (assess imputation performance)
        self.X_org   = tf.placeholder(tf.float32, [None, self.X_dim])
        self.X_u_org = tf.placeholder(tf.float32, [None, self.X_dim])

        # y
        self.y   = tf.placeholder(tf.float32, [None, self.y_dim])
        self.y_g = tf.placeholder(tf.float32, [None, self.y_dim])

        # z
        self.z_G1   = tf.placeholder(tf.float32, [None, self.z_dim])
        self.z_G2   = tf.placeholder(tf.float32, [None, self.z_dim])
        self.z_G2_u = tf.placeholder(tf.float32, [None, self.z_dim])
        self.z_G2_g = tf.placeholder(tf.float32, [None, self.z_dim])

        # m
        self.m   = tf.placeholder(tf.float32, [None, self.X_dim+self.y_dim])
        self.m_u = tf.placeholder(tf.float32, [None, self.X_dim+self.y_dim])
        self.m_g = tf.placeholder(tf.float32, [None, self.X_dim+self.y_dim])

        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        ## Forward pass
        # make H
        H   = self.Encoder(self.X, self.m[:,:-self.y_dim],      self.z_G2, reuse=False)
        H_u = self.Encoder(self.X_u, self.m_u[:,:-self.y_dim],  self.z_G2_u)
        H_g = self.Generator1(self.y_g, self.z_G1, reuse=False)

        # assess H
        D1,   D1_logits   = self.Discriminator1(H, self.y,   reuse=False)
        D1_g, D1_logits_g = self.Discriminator1(H_g, self.y_g)

        # make X^
        X_,   X_G2   = self.Generator2(self.X,   self.m[:,:-self.y_dim],   H,   reuse=False)
        X_u_, X_u_G2 = self.Generator2(self.X_u, self.m_u[:,:-self.y_dim], H_u)
        X_g_, X_g_G2 = self.Generator2(self.m_g[:,:-self.y_dim], self.m_g[:,:-self.y_dim], H_g) # using m_g as dummy X

        # predict y
        self.C, C_logits = self.Classifier(X_,  reuse=False)

        self.C_u, C_logits_u = self.Classifier(X_u_)
        self.C_g, C_logits_g = self.Classifier(X_g_)
        self.pred_u = tf.round(self.C_u)

        # assess X^, y
        D2, D2_logits     = self.Discriminator2(X_,   self.y,      reuse=False)
        D2_u, D2_logits_u = self.Discriminator2(X_u_, self.pred_u)
        D2_g, D2_logits_g = self.Discriminator2(X_g_, self.y_g)

        ## Metrics
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.C, 1), tf.argmax(self.y, 1)), tf.float32))
        self.rmse = RMSE(self.X_org, X_G2, self.m[:,:-self.y_dim], self.X_u_org, X_u_, self.m_u[:,:-self.y_dim])
        self.pr_auc_op, self.pr_auc = tf.metrics.auc(self.y, self.C, curve='PR', summation_method='careful_interpolation')

        ## Losses
        # reconstruction loss
        self.L_recon = MSE(self.X_org, X_G2, self.m[:,:-self.y_dim], self.X_u_org, X_u_G2, self.m_u[:,:-self.y_dim])

        # D&G 1 loss
        L_D1 = tf.reduce_mean(D1_logits_g) - tf.reduce_mean(D1_logits)
        L_G1 = -tf.reduce_mean(D1_logits_g)

        # D&G 2 loss
        D2_Xy = tf.concat([D2_logits, D2_logits_u, D2_logits_g], axis=0)
        D2_m = tf.concat([self.m, self.m_u, self.m_g], axis=0)
        L_G2 = WGAN_G_loss(D2_Xy, D2_m)
        L_D2 = WGAN_D_loss(D2_Xy, D2_m)

        # classifier loss
        L_C = softmax_CE(labels=self.y, logits=C_logits)
        L_C_g = softmax_CE(labels=self.y_g, logits=C_logits_g)
        L_C_lg = softmax_CE(labels=tf.concat([self.y, self.y_g], axis=0), logits=tf.concat([C_logits, C_logits_g], axis=0))

        # trainable variables
        variables = tf.trainable_variables()
        E_vars = [var for var in variables if 'encoder' in var.name]
        G1_vars = [var for var in variables if 'generator1' in var.name]
        D1_vars = [var for var in variables if 'discriminator1' in var.name]
        G2_vars = [var for var in variables if 'generator2' in var.name]
        D2_vars = [var for var in variables if 'discriminator2' in var.name]
        C_vars = [var for var in variables if 'classifier' in var.name]

        # gradient penalty
        real_1_h = H
        real_1_y = self.y
        gp_1, slopes_1 = zc_gradient_penalty_D1(real_1_h, real_1_y, self.Discriminator1)

        real_2_x = tf.concat([X_, X_u_], 0)
        real_2_y = tf.concat([self.y, self.pred_u], 0)
        m_2 = tf.concat([self.m, self.m_u], 0)
        gp_2, slopes_2 = zc_gradient_penalty_D2(real_2_x, real_2_y, m_2, self.Discriminator2)

        # losses for 6 components
        self.E_loss  = L_G2 + 10* self.L_recon
        self.G1_loss = L_G1 + 1 * L_G2 + 0.01 * L_C_g
        self.D1_loss = L_D1 + 10* gp_1
        self.G2_loss = L_G2 + 10* self.L_recon
        self.D2_loss = L_D2 + 10* gp_2
        self.C_loss = L_C_lg + 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in C_vars]) + 0.1 * L_G2

        self.pre_D1_loss = gp_1
        self.pre_D2_loss = gp_2

        ## Optimizer
        # decay
        global_step_GAN = tf.Variable(0, trainable=False)
        global_step_C = tf.Variable(0, trainable=False)
        self.lr_GAN = tf.train.exponential_decay(self.lr_GAN, global_step_GAN, 1, self.decay, staircase=False)
        self.lr_C = tf.train.exponential_decay(self.lr_C, global_step_C, 1, self.decay, staircase=False)
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # recon
            self.pre_E_opt = tf.train.RMSPropOptimizer(self.lr_GAN).minimize(10 * self.L_recon, var_list=E_vars)
            self.pre_G2_opt = tf.train.RMSPropOptimizer(self.lr_GAN).minimize(10 * self.L_recon, var_list=G2_vars)
            # GD1
            self.pre_G1_opt = tf.train.RMSPropOptimizer(self.lr_GAN).minimize(L_G1, var_list=G1_vars)
            self.pre_D1_opt = tf.train.RMSPropOptimizer(10*self.lr_GAN).minimize(self.pre_D1_loss, var_list=D1_vars)
            self.pre_D2_opt = tf.train.RMSPropOptimizer(10*self.lr_GAN).minimize(self.pre_D2_loss, var_list=D2_vars)
            # training
            self.E_opt  = tf.train.RMSPropOptimizer(self.lr_GAN).minimize(self.E_loss,  var_list=E_vars)
            self.G1_opt = tf.train.RMSPropOptimizer(0.1*self.lr_GAN).minimize(self.G1_loss, var_list=G1_vars)
            self.D1_opt = tf.train.RMSPropOptimizer(0.1*self.lr_GAN).minimize(self.D1_loss, var_list=D1_vars)
            self.G2_opt = tf.train.RMSPropOptimizer(self.lr_GAN).minimize(self.G2_loss, var_list=G2_vars, global_step=global_step_GAN)
            self.D2_opt = tf.train.RMSPropOptimizer(self.lr_GAN).minimize(self.D2_loss, var_list=D2_vars)
            self.C_opt  = tf.train.RMSPropOptimizer(self.lr_C).minimize(self.C_loss,  var_list=C_vars, global_step=global_step_C)
