基于LSTM的信号跳变点检测
===================================
## 环境:
Windows

tensorflow:1.8.0

python:3.6

## 开始：
datagen.m 用来生成跳频信号的频谱与标签数据保存在traindata.mat与testdata.mat。

数据集已经上传，可以直接运行：
python FCD_main.py

## 说明：
检测一段采样信号里频率发生跳变的位置。输入LSTM网络的数据为fft后信号的频谱。

训练label为当前时间窗内采样的信号,跳变点位于中点的概率。采用模糊标注策略，即label(n)=max{0，1-|跳变点距中心点的概率|/10}。当跳变点位于采样信号的中心时，label为1，当跳变点离中心店越来越远时，
label减小至0.

输入LSTM的数据格式为(batchsize，timesteps=时域采样点数，1)

![result](https://github.com/Woshiwzl1997/Signal-Frequency-Change-Detection-Based-on-LSTM/blob/master/Figure_1.png)


