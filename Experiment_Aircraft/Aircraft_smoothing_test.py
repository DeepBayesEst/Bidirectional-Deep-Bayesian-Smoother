import tensorflow.compat.v1 as tf                 # 用 compat.v1
tf.disable_v2_behavior()                          # 关闭 eager，回到图模式
import numpy as np
from lstm_tf_filtering import LSTMStateTuple_KF
from lstm_tf_filtering import LSTMCell as LSTMCell_filter
from lstm_tf_smoothing import LSTMCell as LSTMCell_smooth
import random
# from data_prepare import get_batch_data
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from copy import deepcopy

# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# tf.config.threading.set_intra_op_parallelism_threads(3)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 1
timestep_size = 120
dim_state = 4
dim_meas = 2
# meas_noise = 150
azi_n = 0.2*np.pi/180
dis_n = 100
_sT = 4.
# 初始化滤波所需先验信息

def get_total_params_num():
    total_num = 0
    for var in tf.trainable_variables():
        var_shape = var.get_shape()
        var_params_num = 1
        for dim in var_shape:
            var_params_num *= dim.value
        total_num += var_params_num
    return total_num
def plot_heatmaps(m_f_arr, m_b_arr):
    """
    绘制前向和后向数组的热力图对比（自动处理维度不匹配）

    参数:
    m_f_arr (np.ndarray): 前向数组 (119, 64)
    m_b_arr (np.ndarray): 后向数组 (118, 64)
    """
    # 检查并统一维度
    if m_f_arr.shape[0] != m_b_arr.shape[0]:
        min_length = min(m_f_arr.shape[0], m_b_arr.shape[0])
        m_f_arr = m_f_arr[:min_length, :]
        m_b_arr = m_b_arr[:min_length, :]
        print(f"警告：数组长度不一致，已截断为 {min_length} 时间步")

    # 创建图形和布局
    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.suptitle('Forward and Backward Arrays Heatmap Comparison',
                 fontsize=18, fontweight='bold')

    # 使用GridSpec创建布局
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1, 1, 0.05],
                  height_ratios=[1, 1],
                  wspace=0.15, hspace=0.2)

    # 创建子图
    ax1 = fig.add_subplot(gs[0, 0])  # 前向热力图
    ax2 = fig.add_subplot(gs[0, 1])  # 后向热力图
    ax3 = fig.add_subplot(gs[0, 2])  # 颜色条
    ax4 = fig.add_subplot(gs[1, 0])  # 前向行剖面
    ax5 = fig.add_subplot(gs[1, 1])  # 后向行剖面

    # 计算共享的颜色范围
    vmin = min(np.nanmin(m_f_arr), np.nanmin(m_b_arr))
    vmax = max(np.nanmax(m_f_arr), np.nanmax(m_b_arr))

    # 创建自定义颜色映射
    colors = ["blue", "white", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_diverging", colors)

    # 绘制前向热力图
    im1 = ax1.imshow(m_f_arr, cmap=cmap, aspect='auto',
                    vmin=vmin, vmax=vmax,
                    origin='lower', interpolation='none')
    ax1.set_title(f'Forward Array (shape: {m_f_arr.shape})', fontsize=14)
    ax1.set_xlabel('Feature Dimension (0-63)', fontsize=12)
    ax1.set_ylabel('Time Step', fontsize=12)
    ax1.grid(True, linestyle=':', color='gray', alpha=0.3)

    # 绘制后向热力图
    im2 = ax2.imshow(m_b_arr, cmap=cmap, aspect='auto',
                    vmin=vmin, vmax=vmax,
                    origin='lower', interpolation='none')
    ax2.set_title(f'Backward Array (shape: {m_b_arr.shape})', fontsize=14)
    ax2.set_xlabel('Feature Dimension (0-63)', fontsize=12)
    ax2.set_ylabel('Time Step', fontsize=12)
    ax2.grid(True, linestyle=':', color='gray', alpha=0.3)

    # 添加颜色条
    fig.colorbar(im1, cax=ax3)

    # 添加行剖面分析（选择中间时间步）
    time_step = min(m_f_arr.shape[0], m_b_arr.shape[0]) // 2
    ax4.plot(m_f_arr[time_step, :], 'b-o', linewidth=1.5, markersize=4)
    ax4.set_title(f'Feature Profile at Time Step {time_step} (Forward)', fontsize=12)
    ax4.set_xlabel('Feature Index', fontsize=10)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.grid(True, alpha=0.3)

    ax5.plot(m_b_arr[time_step, :], 'r-o', linewidth=1.5, markersize=4)
    ax5.set_title(f'Feature Profile at Time Step {time_step} (Backward)', fontsize=12)
    ax5.set_xlabel('Feature Index', fontsize=10)
    ax5.set_ylabel('Value', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = (
        f"Forward Array Stats:\n"
        f"Shape: {m_f_arr.shape}\n"
        f"Min: {np.min(m_f_arr):.4f}\n"
        f"Max: {np.max(m_f_arr):.4f}\n"
        f"Mean: {np.mean(m_f_arr):.4f}\n"
        f"Std: {np.std(m_f_arr):.4f}\n\n"
        f"Backward Array Stats:\n"
        f"Shape: {m_b_arr.shape}\n"
        f"Min: {np.min(m_b_arr):.4f}\n"
        f"Max: {np.max(m_b_arr):.4f}\n"
        f"Mean: {np.mean(m_b_arr):.4f}\n"
        f"Std: {np.std(m_b_arr):.4f}"
    )

    fig.text(0.92, 0.5, stats_text, fontsize=11,
             bbox=dict(facecolor='whitesmoke', alpha=0.7),
             verticalalignment='center')

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    # plt.savefig('heatmap_comparison.png', bbox_inches='tight', dpi=150)
    plt.show()

def F_cv(bt):
    temp0 = [1., 0., _sT, 0.]
    temp1 = [0., 1., 0., _sT]
    temp2 = [0., 0., 1., 0.]
    temp3 = [0., 0., 0., 1.]
    F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
    F_c = tf.cast(F_c, tf.float64)
    F = tf.stack([F_c for _ in range(bt)])
    return F

# 雷达量测矩阵
def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    x = tf.reshape(x,[batch_size,dim_state])
    r0= tf.atan2(x[:,1],x[:,0])
    r1= tf.sqrt(tf.square(x[:,0])+tf.square(x[:,1]))
    r = tf.stack([r0,r1],axis = 1)
    return tf.reshape(r,[batch_size,dim_meas,1])

# 量测雅可比矩阵
def hx(x_):
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
    temp0 = [1., 0., 0., 0.]
    temp1 = [0., 1., 0., 0.]
    H_ = tf.stack([temp0, temp1], axis=0)
    H_ = tf.cast(H_, tf.float64)
    H_bt = tf.stack([H_ for _ in range(bt)])
    return H_bt

def Meas_noise(bt):
    R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise

def Pro_noise(bt):
    Q_mat = np.diag([100,100,100,100])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise

F_bt = F_cv(batch_size)
H_bt = H_linear(batch_size)
R_bt = Meas_noise(batch_size)
Q_bt = Pro_noise(batch_size)

H_J = hx
h = my_hx

# 定义网络
m_hidden = 64

# 定义全局步数变量
with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)

with tf.name_scope("globa_lstep"):
    global_steps2 = tf.Variable(0, trainable=False)

# 通过TensorFlow中的placeholder定义输入占位
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,4))
    h_start = tf.placeholder(tf.float64, shape=(None,4))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))
    # Q_filt = tf.placeholder(tf.float64, shape=(None,timestep_size-1,4,4))
    # Q_smoo = tf.placeholder(tf.float64, shape=(None,timestep_size-2,4,4))

cell_filter = LSTMCell_filter(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

cell_smooth = LSTMCell_smooth(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))

state_m = initial_state_m


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

loss11 = tf.reduce_mean(tf.square(y_-M_arr[:,:,:]))
loss0 = tf.reduce_mean(tf.square(y_[0]-M_arr[0,:,:]))
loss_pos = tf.reduce_mean(tf.square(y_[:,:,:2]-M_arr[:,:,:2]))

smooth_M_arr = tf.transpose(tf.stack(smooth_M_lst), [1, 0, 2])

smooth_loss11 = tf.reduce_mean(tf.square(y_-smooth_M_arr))

smooth_loss0 = tf.reduce_mean(tf.square(y_[0]-smooth_M_arr[0,:]))
smooth_loss_pos = tf.reduce_mean(tf.square(y_[:,:,:2]-smooth_M_arr[:,:,:2]))

loss_all = loss11 + smooth_loss11

loss_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w_train1')
print(loss_vars1)

loss_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='w_train2')
print(loss_vars2)

saver = tf.train.Saver()

# 读取数据
data_num = 800 # 数据规模

import time
data_set = np.load('true_smooth_air_120.npy')[5000:5000+data_num]

with tf.Session() as sess:
    para_num = get_total_params_num()
    print('para_num:', para_num)
    saver.restore(sess, './Air/model.ckpt')
    print('开始运行测试...')
    for mc in range(100):
        loss_filtering_list = []
        loss_smoothing_list = []
        for num in range(320):
            data = data_set[num, :, :].reshape([1,timestep_size,4])
            state_batch = data[:,:timestep_size,:].reshape([batch_size,timestep_size,4])
            meas_batch = deepcopy(data[:, :timestep_size, :2].reshape([batch_size, timestep_size, 2]))
            measurement = np.zeros_like(meas_batch)
            Obser = np.zeros_like(meas_batch)
            for i in range(batch_size):
                measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + np.random.normal(0, azi_n,
                                                                                                               timestep_size)
                measurement[i, :, 1] = np.sqrt(
                    np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + np.random.normal(0, dis_n,
                                                                                                        timestep_size)  #
                Obser[i, :, 0] = measurement[i, :, 1] * np.cos(measurement[i, :, 0])
                Obser[i, :, 1] = measurement[i, :, 1] * np.sin(measurement[i, :, 0])
            h_start_in = np.zeros([batch_size, dim_state], dtype='float64')
            h_start_in[:, 0] = state_batch[:, 0, 0]
            h_start_in[:, 1] = state_batch[:, 0, 1]
            h_start_in[:, 2] = state_batch[:, 0, 2] #
            h_start_in[:, 3] = state_batch[:, 0, 3] #
            c_start_ = np.zeros([batch_size, m_hidden])
            P_start_ = np.stack(np.array([1000*np.eye(4) for _ in range(batch_size)], "float64"))
            start_t = time.time()
            loss_filtering,filter_arr,smooth_arr,loss_0,loss_pos_,\
                loss_smoothing = sess.run([loss11, M_arr,smooth_M_arr,loss0,
                                           loss_pos, smooth_loss11],
                                          feed_dict={x:measurement, y_: state_batch[:, 1:, :],
                                                     h_start:h_start_in, c_start:c_start_,
                                                     P_start:P_start_})


            end_t = time.time()
            loss_filtering_list.append(np.sqrt(loss_filtering))
            loss_smoothing_list.append(np.sqrt(loss_smoothing))

        print('**************************************************')
        print('Step %d'%num) #
        loss_filtering_array = np.array(loss_filtering_list)
        loss_smoothing_array = np.array(loss_smoothing_list)
        print('Loss filtering：',np.mean(loss_filtering_array))
        print('Loss smoothing：', np.mean(loss_smoothing_array))
        print('Measurement loss:',np.sqrt(np.mean(np.square(state_batch[:,:,:2]-Obser))))
        loss_filtering_list = []
        loss_smoothing_list = []
        print('**************************************************')


