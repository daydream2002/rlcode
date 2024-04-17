#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/4
# @Author  : xiu
# @File    : load_model.py
# @Description: 模型加载
import os
import sys
import numpy as np
from reward.extract import get_features, get_features1

sys.path.append('/home/tonnn/.nas/.xiu/works/node4-shangrao_mj_rl_v4_suphx/')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def batch_gen(x, y, round, batch_size):
    """
    数据生成器
    Args:
        x: 特征
        y: 标签
        round: 回合数
        batch_size: 批次大小
    """
    batches = int(x.shape[0] // batch_size)
    for i in range(batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size]
        batch_y = y[i * batch_size:(i + 1) * batch_size]
        batch_r = round[i * batch_size:(i + 1) * batch_size]
        features = np.zeros((batch_size, 26, 43316))
        for j in range(batch_size):
            features[j, :batch_r[j]] = [feature[0] for feature in get_features(batch_x[j])]
        yield features, batch_y


def generator(state, game):
    """
    数据生成器
    Args:
        state: 状态
        game: 游戏类

    Returns: 特征矩阵

    """
    # features = np.zeros((1, 26, 43316))
    # features[0, :1] = [get_features1(state, game)[0]]
    feature = get_features1(state, game)[0]
    feature = feature[np.newaxis, :]
    return feature

# # 数据集json文件(80788局游戏数据)
# with open('./reward/high_score.json', 'r', encoding='utf-8')as f:
#     data = json.load(f)
# # 回合数(80788,)，最大回合数26
# with open('./reward/rounds.json', 'r', encoding='utf-8')as f:
#     rounds = json.load(f)
# # 标签(80788,),最低值-528，最高值1576
# with open('./reward/scores.json', 'r', encoding='utf-8')as f:
#     labels = json.load(f)
#
# tra_val_size = 75600
# x_test = np.array(data[tra_val_size:])
# y_test = np.array(labels[tra_val_size:])
# rounds_test = np.array(rounds[tra_val_size:])
#
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     saver = tf.compat.v1.train.import_meta_graph("./multi_gru_saved/model_ckpt.meta")
#     saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./multi_gru_saved/'))
#     graph = tf.compat.v1.get_default_graph()
#     # 加载模型中的操作节点
#     self_acc = graph.get_tensor_by_name('accuracy:0')
#     self_scores = graph.get_tensor_by_name('scores:0')
#     self_x = graph.get_tensor_by_name('inputs/x:0')
#     self_y = graph.get_tensor_by_name('inputs/y:0')
#     # w = graph.get_tensor_by_name('scores/W:0')
#     # b = graph.get_tensor_by_name('scores/b:0')
#     # gen_batch_test = batch_gen(x_test, y_test, rounds_test, 128)
#     gen_batch_test = batch_gen(x_test, y_test, rounds_test, 10)
#
#     for i in range(5):
#         x_batch_test, y_batch_test = next(gen_batch_test)
#         y_batch_test = y_batch_test[:, np.newaxis]
#         # _w = sess.run(w, feed_dict={self_x: x_batch_test, self_y: y_batch_test})
#         # _b = sess.run(b, feed_dict={self_x: x_batch_test, self_y: y_batch_test})
#         accuracy = sess.run(self_acc, feed_dict={self_x: x_batch_test, self_y: y_batch_test})
#         scores = sess.run(self_scores, feed_dict={self_x: x_batch_test, self_y: y_batch_test})
#         # print("_w:", _w)
#         # print("_b:", _b)
#         print("Scores: ", scores)
#         print("y: ", y_batch_test)
#         print("Test Accuracy: ", "{:.4f}".format(accuracy))
