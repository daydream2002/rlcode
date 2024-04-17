#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : plotjson.py
# @Description: 图像绘制
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# # 设置中文显示
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False

def plotTrainImg(jsonPath, imgTitle, imgSavePath, startNup=0, endNup=-1, xlabel="n_updates", ylabel="indexs"):
    '''

    :param jsonPath: json文件存放的路径， str格式
    :param imgTitle: 图片的title， str 英文
    :param startNup: 绘画时开始的位置，默认为0 int
    :param endNup:   绘画结束时第n个update，默认为-1
    :param imgSavePath: 图片保存的路径
    :return: None
    '''
    json_data = pd.read_json(jsonPath, lines=True)
    # print(json_data.columns)

    # 设置图像大小及清晰度
    plt.figure(figsize=(20, 8), dpi=400)

    # 绘制图像
    plt.plot(json_data["n_updates"][startNup:endNup], json_data["ep_reward_mean"][startNup:endNup],
             label="ep_reward_mean", color="blue")
    plt.plot(json_data["n_updates"][startNup:endNup], json_data["ep_len_mean"][startNup:endNup] / 10,
             label="ep_len_mean/10", color="red")

    plt.plot(json_data["n_updates"][startNup:endNup], json_data["approxkl"][startNup:endNup] / 100, label="approxkl",
             color="gold")
    plt.plot(json_data["n_updates"][startNup:endNup], json_data["ep_win_rate"][startNup:endNup], label="ep_win_rate",
             color="black")

    plt.plot(json_data["n_updates"][startNup:endNup], json_data["policy_entropy"][startNup:endNup],
             label="policy_entropy", color="cyan")
    plt.plot(json_data["n_updates"][startNup:endNup], json_data["policy_loss"][startNup:endNup], label="policy_loss",
             color="pink")
    plt.plot(json_data["n_updates"][startNup:endNup], json_data["value_loss"][startNup:endNup], label="value_loss",
             color="green")
    # 设置坐标轴刻度
    # plt.xticks(x_ticks, x_ticks_label[::5])
    # plt.yticks(y_ticks[::5])

    # 设置网格
    plt.grid(True, linestyle="--", alpha=1)

    # 设置坐标轴含义
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(imgTitle, fontsize=20)

    # 添加图例
    plt.legend(loc="best", fontsize=16)

    # # 图像保存
    plt.savefig(imgSavePath)
    # 展示图像
    # plt.show()


if __name__ == '__main__':
    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.95  # 折扣系数
    # ecf = 0  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = "no"  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-18-10-45/"
    # no_ = "9"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.95  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.6
    # mid = "no"  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-19-15-39/"
    # no_ = "6"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-16-15-25/"
    # no_ = "3"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.99  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # no_ = "5"

    ########### ____train_with_raw____  21.1.22
    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_with_raw/train_log/SAVETIME-2021-01-22-10-26/"
    # no_ = "10"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.6
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_with_raw/train_log/SAVETIME-2021-01-22-11-06/"
    # no_ = "4"

    # startNup = 10  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res34"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_with_raw/train_log/SAVETIME-2021-01-26-15-57/"
    # no_ = "2"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-28-11-28/"
    # basePath = "./train_with_raw/train_log/SAVETIME-2021-01-28-11-28/"
    # no_ = "sqrtR_1"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = 2000  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = "0.01"  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res101"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-28-11-28/"
    # basePath = "./train_with_raw/train_log/SAVETIME-2021-02-24-11-27/"
    # no_ = "sqrtR_1"

    ########### ____train_with_raw____ end

    ########### ____train_with_preModel____
    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.5
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-22-14-11/"
    # no_ = "3"

    # startNup = 0  # 开始绘画的nupdate
    # endNup = -1  # 结束绘画的nupdata
    # lr = 3e-4  # 学习率
    # gm = 0.925  # 折扣系数
    # ecf = 0.01  # 熵函数折扣系数
    # vfcoef = 0.6
    # mid = ""  # 是否有中间步
    # gang = "no"  # 是否有杠分
    # abort = -1  # 流局的reward
    # policy = "res18"  # 策略
    #
    # # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-13-16-20/"
    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-22-20-55/"
    # no_ = "2"

    startNup = 0  # 开始绘画的nupdate
    endNup = 2000  # 结束绘画的nupdata
    lr = 3e-4  # 学习率
    gm = 0.925  # 折扣系数
    ecf = 0.01  # 熵函数折扣系数
    vfcoef = 0.5
    mid = "0.01"  # 是否有中间步
    gang = "no"  # 是否有杠分
    abort = -1  # 流局的reward
    policy = "res101"  # 策略

    # basePath = "./train_from_pretrain/train_log/SAVETIME-2021-01-28-11-28/"
    # basePath = "./train_with_raw/train_log/SAVETIME-2021-03-04-21-59/"
    # basePath = "./train_from_saveModel/train_log/SAVETIME-2021-04-12-22-55/"
    # basePath = "./train_from_saveModel/train_log/SAVETIME-2021-04-12-23-10/"
    basePath = "./"
    no_ = "sqrtR_1"

    jsonPath = basePath + "progress.json"

    imgTitle = "ppo2_our_entCoef0.005_Analysis_of_each_index"

    imgName = imgTitle + "_start" + str(startNup) + "_end" + str(endNup) + "_" + no_ + ".png"
    imgSavePath = basePath + imgName

    plotTrainImg(jsonPath=jsonPath, imgTitle=imgTitle, imgSavePath=imgSavePath, startNup=startNup, endNup=endNup)
