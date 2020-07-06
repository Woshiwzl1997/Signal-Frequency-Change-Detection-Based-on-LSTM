clc
clear all
N_t=64;%滑动窗长
fre_N=200;%频率集点数
freset=10e5:(10e5/fre_N):20e5;%频率集
fs=40e5;%采样频率

fm=1e3;%调制频率
train_num=10000;
%% 构造训练数据集
train_lab=[];
train_sig=[];
tao=10;
show=1;
for n=1:train_num
    %构造跳变数据
    split_posi=randi([2,N_t-2],1);%跳频点
    tl=(0:split_posi-1)*(1/fs);%l左
    tr=(split_posi:N_t-1)*(1/fs);%r右

    f=randperm(fre_N, 2);%随机在频率集里选择两个频率
    
    cur_label=max(0,1-abs(split_posi-N_t/2)/tao);
    train_lab=[train_lab;cur_label];
    signall=exp(1i*2*pi*(freset(f(1))+randi([1,4],1)*fm)*tl);%构造信号
    signalr=exp(1i*2*pi*(freset(f(2))+randi([1,4],1)*fm)*tr);
    cur_signal=[signall signalr];
    spec=abs(fft(cur_signal,N_t));
    spec=mapminmax(spec,0,1);
    train_sig=[train_sig;spec];

    %%画图
    if show==1
       freset(f(1))
       freset(f(2))
       ff=(0:N_t-1)*(fs/N_t);
       figure()
       plot(ff,spec);
       show=0;
    end
end
%构造单频数据
for n=1:train_num/10
    t=(0:N_t-1)*(1/fs);
    ff=randperm(fre_N, 1);
    train_lab=[train_lab;0];
    signal=exp(1i*2*pi*(freset(ff)+randi([1,4],1)*fm)*t);
    spec=abs(fft(signal,N_t));
    spec=mapminmax(spec,0,1);%归一化
    train_sig=[train_sig;spec];
end
save('traindata.mat','train_sig','train_lab')
%% 构造测试数据
test_fre=[12e5,18e5,13e5,20e5];
test_lab=[];
test_sig=[];
for n=1:length(test_fre)-1
    %单频
    test_lab=[test_lab;0];
    signal=exp(1i*2*pi*(test_fre(n)+randi([1,4],1)*fm)*t);
    spec=abs(fft(signal,N_t));
    spec=mapminmax(spec,0,1);
    test_sig=[test_sig;spec];
    %跳频
    for hop=2:N_t-2
        split_posi=hop;%跳频点
        tl=(0:split_posi-1)*(1/fs);
        tr=(split_posi:N_t-1)*(1/fs);
        
        cur_label=max(0,1-abs(split_posi-N_t/2)/tao);
        test_lab=[test_lab;cur_label];
        
        signall=exp(1i*2*pi*(test_fre(n)+randi([1,4],1)*fm)*tl);%构造信号
        signalr=exp(1i*2*pi*(test_fre(n+1)+randi([1,4],1)*fm)*tr);
        
        cur_signal=[signall signalr];
        spec=abs(fft(cur_signal,N_t));
        spec=mapminmax(spec,0,1);
        
        test_sig=[test_sig;spec];
        if hop==N_t/2
           ff=(0:N_t-1)*(fs/N_t);
           figure()
           plot(ff,spec);
        end
    end
end
save('testdata.mat','test_sig','test_lab')




