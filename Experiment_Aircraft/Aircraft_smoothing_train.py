
import tensorflow.compat.v1 as tf                 # if tensorflow 2.X
tf.disable_v2_behavior()                          # Disable eager execution; use static computation graph mode

# import tensorflow as tf                 # if tensorflow 1.X

import os
import numpy as np
from lstm_tf_filtering import LSTMStateTuple_KF
from lstm_tf_filtering import LSTMCell as LSTMCell_filter
from lstm_tf_smoothing import LSTMCell as LSTMCell_smooth

from copy import deepcopy
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU selection

# Parameters
batch_size = 8
timestep_size = 120
dim_state = 4
dim_meas = 2
# Measurement noise
azi_n = 0.2*np.pi/180
dis_n = 100
# Sample time
_sT = 4.
# number of hidden layer nodes
m_hidden = 64

def safe_inv(P, epsilon=1e-3):
    """
    Safe matrix inverse with diagonal regularization to avoid singularity
    """
    I = tf.eye(tf.shape(P)[-1], dtype=P.dtype)
    return tf.linalg.inv(P + epsilon * I)
def nll_loss(y_true, y_pred, P):
    '''
    Negative Log Likelihood loss using multivariate Gaussian form
    Args:
        y_true:
        y_pred:
        P: Estimated covariance
    Returns: NLL loss
    '''
    residual = y_true - y_pred
    # dim = tf.cast(tf.shape(residual)[-1], tf.float64)
    term1 = 0.5 * tf.squeeze(tf.matmul(
        tf.expand_dims(residual, -2),
        tf.matmul(safe_inv(P), tf.expand_dims(residual, -1))
    ), [-1, -2])
    term2 = 0.5 * tf.linalg.logdet(P)
    return term1 + term2

def get_total_params_num():
    '''
    Total parameter number
    '''
    total_num = 0
    for var in tf.trainable_variables():
        var_shape = var.get_shape()
        var_params_num = 1
        for dim in var_shape:
            var_params_num *= dim.value
        total_num += var_params_num
    return total_num

def F_cv(bt):
    """Constant velocity motion model F matrix (shared across batch)"""
    temp0 = [1., 0., _sT, 0.]
    temp1 = [0., 1., 0., _sT]
    temp2 = [0., 0., 1., 0.]
    temp3 = [0., 0., 0., 1.]
    F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
    F_c = tf.cast(F_c, tf.float64)
    F = tf.stack([F_c for _ in range(bt)])
    return F

def my_hx(x):
    """Nonlinear measurement function: converts Cartesian [x, y] to [azimuth, range]"""
    x = tf.reshape(x,[batch_size,dim_state])
    r0= tf.atan2(x[:,1],x[:,0])
    r1= tf.sqrt(tf.square(x[:,0])+tf.square(x[:,1]))
    r = tf.stack([r0,r1],axis = 1)
    return tf.reshape(r,[batch_size,dim_meas,1])

def hx(x_):
    """Jacobian of the measurement function"""
    x_ = tf.reshape(x_,[batch_size,dim_state])
    for i in range(batch_size):
        x = x_[i]
        temph0 = [-x[1] / (tf.square(x[0]) + tf.square(x[1])),
                  x[0] / (tf.square(x[0]) + tf.square(x[1])), 0., 0.]
        temph1 = [x[0] / tf.sqrt((tf.square(x[0]) + tf.square(x[1]))),
                  x[1] / tf.sqrt((tf.square(x[0]) + tf.square(x[1]))), 0., 0.]
        H_mat = tf.stack([temph0, temph1], axis=0)
        if i == 0:
            H_out = H_mat
        else:
            H_out = tf.concat([H_out,H_mat],axis = 0)
    H_out = tf.reshape(H_out,[batch_size,2,4])
    # tf.reshape(tf.matmul(H_out, tf.reshape(x_, [batch_size, 4, 1])), [batch_size, 2]),
    return H_out

def H_linear(bt):
    """(Unused) Linear measurement matrix"""
    temp0 = [1., 0., 0., 0.]
    temp1 = [0., 1., 0., 0.]
    H_ = tf.stack([temp0, temp1], axis=0)
    H_ = tf.cast(H_, tf.float64)
    H_bt = tf.stack([H_ for _ in range(bt)])
    return H_bt

def Meas_noise(bt):
    """Measurement noise covariance R"""
    R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise

def Pro_noise(bt):
    """Process noise covariance Q"""
    Q_mat = np.diag([36,36,36,36])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise

F_bt = F_cv(batch_size)
H_bt = H_linear(batch_size)
R_bt = Meas_noise(batch_size)
Q_bt = Pro_noise(batch_size)

H_J = hx
h = my_hx

with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)

with tf.name_scope("globa_lstep"):
    global_steps2 = tf.Variable(0, trainable=False)

# 通过TensorFlow中的placeholder定义输入占位
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))# Measurements
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,4))# Ground truth state
    h_start = tf.placeholder(tf.float64, shape=(None,4))# Initial state
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))# Initial memory
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))# Initial covariance

# Model Definitions
cell_filter = LSTMCell_filter(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

cell_smooth = LSTMCell_smooth(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))

'''
Forward Filtering Pass
'''
state_m = initial_state_m
M_lst = []
smooth_M_lst = []
m_p_lst = []
P_pred = []
P_filtered = []
for time_step in range(1,timestep_size):
    print('forward',time_step)
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m, m_p, P_f ,P_p) = cell_filter(x[:, time_step, :],state_m)
        # Storing some intermediate and result variables
        M_lst.append(pred)
        m_p_lst.append(m_p)
        P_pred.append(P_p)
        P_filtered.append(state_m[2])

# filtered result
M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])

'''
Backward Smoothing Pass
'''
C_filter,_,_ = state_m
smooth_M_lst = M_lst
P_smoothed = []
P_smoothed.append(tf.reshape(state_m[2],[batch_size,4,4]))

for backtime in range(timestep_size-3,-1,-1):
    print('backward',backtime)
    with tf.variable_scope('w_train2', reuse=tf.AUTO_REUSE):
        input_smooth = tf.concat([
            smooth_M_lst[backtime + 1],
            M_lst[backtime],
            tf.reshape(m_p_lst[backtime + 1], [batch_size, dim_state]),
            P_filtered[backtime],
            tf.reshape(P_pred[backtime + 1], [batch_size, dim_state * dim_state]),
            C_filter], axis=1)

        (smooth_m, state_m) = cell_smooth(input_smooth,state_m)
        # Storing some intermediate and result variables
        P_smoothed.append(state_m[2])
        smooth_M_lst[backtime] = smooth_m

P_filtered_arr = tf.reshape(tf.transpose(tf.stack(P_filtered),[1,0,2]),[batch_size,timestep_size-1,4,4])
P_smoothed_arr = tf.transpose(tf.stack(P_smoothed),[1,0,2,3])

# smoothed result
smooth_M_arr = tf.transpose(tf.stack(smooth_M_lst), [1, 0, 2])

'''Calculating the loss function'''
# Filtering loss
filter_dist = nll_loss(y_, M_arr, P_filtered_arr)
loss_filter = tf.reduce_mean(filter_dist)
# Global loss
smooth_dist = nll_loss(y_[:, :, :], smooth_M_arr[:, :, :], P_smoothed_arr)
loss_smooth = tf.reduce_mean(smooth_dist)

# Setting two-stage learning rate and decay
learing_rate1 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=500,
                                          decay_rate=0.9)

learning_rate2 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=500,
                                          decay_rate=0.9)
# Partitioning learnable parameters
loss_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w_train1')
loss_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w_train2')

# Define two-stage training steps
train_step1 = tf.train.AdamOptimizer(learing_rate1).minimize(loss_filter, global_step=global_steps1, var_list=loss_vars1)
train_step2 = tf.train.AdamOptimizer(learning_rate2).minimize(loss_smooth, global_step=global_steps2)

saver = tf.train.Saver()

# Reading Data
data_set = np.load('true_smooth_air_120.npy')[:5000]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Start training...')
    para_num = get_total_params_num()
    print('Learnable parameter number:', para_num)
    loss_filtering_list = []
    loss_smoothing_list = []
    # Training
    for step in range(20000):
        a = np.array([i for i in range(data_set.shape[0])])
        b = np.random.choice(a, size=batch_size)
        data = data_set[b]
        state_batch = data[:,:timestep_size,:].reshape([batch_size,timestep_size,4])
        # Generate polar coordinate measurements
        meas_batch = deepcopy(data[:, :timestep_size, :2].reshape([batch_size, timestep_size, 2]))
        measurement = np.zeros_like(meas_batch)
        Obser = np.zeros_like(meas_batch)

        for i in range(batch_size):
            # Azimuth
            measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + np.random.normal(0, azi_n,
                                                                                                           timestep_size)
            # Radial distance
            measurement[i, :, 1] = np.sqrt(
                np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + np.random.normal(0, dis_n,
                                                                                                    timestep_size)  #
            # Convert back to Cartesian coordinate system for easy viewing
            Obser[i, :, 0] = measurement[i, :, 1] * np.cos(measurement[i, :, 0])
            Obser[i, :, 1] = measurement[i, :, 1] * np.sin(measurement[i, :, 0])

        # Initialization
        h_start_in = np.zeros([batch_size, dim_state], dtype='float64')
        h_start_in[:, 0] = state_batch[:, 0, 0]
        h_start_in[:, 1] = state_batch[:, 0, 1]
        h_start_in[:, 2] = state_batch[:, 0, 2] #
        h_start_in[:, 3] = state_batch[:, 0, 3] #
        c_start_ = np.zeros([batch_size, m_hidden])
        P_start_ = np.stack(np.array([1*np.eye(4) for _ in range(batch_size)], "float64"))

        # Execution of training
        _,loss_filtering = sess.run([train_step1,loss_filter],feed_dict={x:measurement,y_:state_batch[:,1:,:],h_start:h_start_in,c_start:c_start_,P_start:P_start_})

        _, loss_smoothing = sess.run([train_step2, loss_smooth],feed_dict={x:measurement,y_:state_batch[:,1:,:],h_start:h_start_in,c_start:c_start_,P_start:P_start_})

        loss_filtering_list.append(np.sqrt(loss_filtering))
        loss_smoothing_list.append(np.sqrt(loss_smoothing))
        if step % 100 == 0:
            loss_filtering_array = np.array(loss_filtering_list)
            loss_smoothing_array = np.array(loss_smoothing_list)
            print('**************************************************')
            print('%d steps have been run'%step)
            print('Filter Loss：',np.mean(loss_filtering_array))
            print('Smooth Loss：', np.mean(loss_smoothing_array))
            print('Measurement noise:',np.sqrt(np.mean(np.square(state_batch[:,:,:2]-Obser))))
            loss_filtering_list = []
            loss_smoothing_list = []
            print('**************************************************')
        # if step % 1000 == 0:
        #     saver.save(sess, "./Air/model.ckpt")
