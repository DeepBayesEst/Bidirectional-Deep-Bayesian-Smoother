# import tensorflow as tf
import tensorflow.compat.v1 as tf                 #
tf.disable_v2_behavior()                          #
import numpy as np
from lstm_tf_filtering_car import LSTMStateTuple_KF
from lstm_tf_filtering_car import LSTMCell as LSTMCell_filter
from lstm_tf_smoothing_car import LSTMCell as LSTMCell_smooth
import matplotlib.pyplot as plt
import os
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size_t = 20
batch_size = 20
timestep_size = 60
dim_state = 4
dim_meas = 2
_sT = 1.

def get_total_params_num():
    total_num = 0
    for var in tf.trainable_variables():
        var_shape = var.get_shape()
        var_params_num = 1
        for dim in var_shape:
            var_params_num *= dim.value
        total_num += var_params_num
    return total_num

def F_cv(bt):
    temp0 = [1., 0., _sT, 0.]
    temp1 = [0., 1., 0., _sT]
    temp2 = [0., 0., 1., 0.]
    temp3 = [0., 0., 0., 1.]
    F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
    F_c = tf.cast(F_c, tf.float64)
    F = tf.stack([F_c for _ in range(bt)])
    return F

def H_linear(bt):
    temp0 = [0., 0., 1., 0.]
    temp1 = [0., 0., 0., 1.]
    H_ = tf.stack([temp0, temp1], axis=0)
    H_ = tf.cast(H_, tf.float64)
    H_bt = tf.stack([H_ for _ in range(bt)])
    return H_bt

def Meas_noise(bt):
    R_mat = np.array([[np.square(0.1), 0], [0, np.square(0.1)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise

def Pro_noise(bt):
    Q_mat = np.diag([1,1,1,1])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise

F_bt = F_cv(batch_size)
H_bt = H_linear(batch_size)
R_bt = Meas_noise(batch_size)
Q_bt = Pro_noise(batch_size)

m_hidden = 64

with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)

with tf.name_scope("globa_lstep"):
    global_steps2 = tf.Variable(0, trainable=False)

with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,4))
    h_start = tf.placeholder(tf.float64, shape=(None,4))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))

cell_filter = LSTMCell_filter(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = H_bt,meas_matrix = H_bt,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

cell_smooth = LSTMCell_smooth(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = H_bt,meas_matrix = H_bt,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))

state_m = initial_state_m


M_lst = []
smooth_M_lst = []
m_p_lst = []
P_pred = []
P_filtered = []

for time_step in range(1,timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m, m_p, P_f ,P_p) = cell_filter(x[:, time_step, :],state_m)
        M_lst.append(pred)
        m_p_lst.append(m_p)
        P_pred.append(P_p)
        P_filtered.append(P_f)

C_filter,_,_=state_m
M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])
loss11 = tf.reduce_mean(tf.square(y_-M_arr[:,:,:]))
loss0 = tf.reduce_mean(tf.square(y_[0]-M_arr[0,:,:]))
loss_pos = tf.reduce_mean(tf.square(y_[:,:,:2]-M_arr[:,:,:2]))
smooth_M_lst = M_lst
state_ms = state_m

for backtime in range(timestep_size-3,-1,-1):
    with tf.variable_scope('w_train2', reuse=tf.AUTO_REUSE):

        input_smooth = tf.concat([
            smooth_M_lst[backtime + 1],
            M_lst[backtime],
            tf.reshape(m_p_lst[backtime + 1], [batch_size, dim_state]),
            P_filtered[backtime],
            tf.reshape(P_pred[backtime + 1], [batch_size, dim_state * dim_state]),
            C_filter
        ], axis=1)

        (smooth_m, state_ms) = cell_smooth(input_smooth,state_ms)

        smooth_M_lst[backtime] = smooth_m

smooth_M_arr = tf.transpose(tf.stack(smooth_M_lst), [1, 0, 2])
smooth_loss11 = tf.reduce_mean(tf.square(y_-smooth_M_arr))
smooth_loss0 = tf.reduce_mean(tf.square(y_[0]-smooth_M_arr[0,:]))
smooth_loss_pos = tf.reduce_mean(tf.square(y_[:,:,:2]-smooth_M_arr[:,:,:2]))
loss_s_output = tf.sqrt(tf.reduce_mean(tf.square(((y_[:, :, 0] - smooth_M_arr[:, :, 0])**2+(y_[:, :, 1] - smooth_M_arr[:, :, 1])**2)**0.5)))
loss_all = loss11 + smooth_loss11

#
learing_rate1 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=2000,
                                          decay_rate=0.5)

learning_rate2 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=2000,
                                          decay_rate=0.5)

loss_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w_train1')
# print(loss_vars1)

loss_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w_train2')
# print(loss_vars2)

lambda_l2 = 0.001

l2_loss_vars1 = tf.add_n([tf.nn.l2_loss(v) for v in loss_vars1]) * lambda_l2
l2_loss_vars2 = tf.add_n([tf.nn.l2_loss(v) for v in loss_vars2]) * lambda_l2


total_loss1 = loss_pos * 10 + loss11 + l2_loss_vars1
total_loss2 = smooth_loss_pos * 10 + smooth_loss11 + l2_loss_vars2

#
train_step1 = tf.train.AdamOptimizer(learing_rate1).minimize(loss_pos*10+loss11, global_step=global_steps1, var_list=loss_vars1)
train_step2 = tf.train.AdamOptimizer(learning_rate2).minimize(smooth_loss_pos*10+smooth_loss11, global_step=global_steps2)

saver = tf.train.Saver()

# 读取数据
data_ture = np.load('data_true_60.npy')[20:,:,:]
data_meas = np.load('data_meas_60.npy')[20:,:,:]
test_data_ture = np.load('data_true_60.npy')[:20,:,:]
test_data_meas = np.load('data_meas_60.npy')[:20,:,:]
with tf.Session() as sess:
    para_num = get_total_params_num()
    print('para_num:', para_num)
    tf.global_variables_initializer().run()

    test_state_batch = test_data_ture[:, :timestep_size, :].reshape([batch_size_t, timestep_size, 4])
    test_meas_batch = test_data_meas

    test_h_start_in = np.zeros([batch_size_t, dim_state], dtype='float64')
    test_h_start_in[:, 0] = test_state_batch[:, 0, 0]
    test_h_start_in[:, 1] = test_state_batch[:, 0, 1]
    test_h_start_in[:, 2] = test_state_batch[:, 0, 2]
    test_h_start_in[:, 3] = test_state_batch[:, 0, 3]

    test_c_start_ = np.zeros([batch_size_t, m_hidden])
    test_P_start_ = np.stack(100 * np.array([np.eye(4) for _ in range(batch_size_t)], "float64"))

    # 执行测试
    test_loss, test_resu, test_pos_loss,loss_s_output_ = sess.run([smooth_loss11, M_arr, smooth_loss_pos,loss_s_output],
                                                   feed_dict={x: test_meas_batch,
                                                              y_: test_state_batch[:, 1:, :],
                                                              h_start: test_h_start_in, c_start: test_c_start_,
                                                              P_start: test_P_start_})
    print('Test loss：', np.sqrt(test_loss))
    print('Test pos loss:', np.sqrt(test_pos_loss))
    print('smooth pos',loss_s_output_)

