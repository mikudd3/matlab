clc % 清屏
clear all; % 删除workplace变量
close all; % 关掉显示图形窗口
format short
% Initial
%% 训练数据预测数据提取及归一化


%输出数据
output1 = train_label;
output2 = test_label;
%把输出从1维变成3维
for i=1:m1
    switch output1(i)
        case 1
            output11(i,:)=[1 0 0];
        case 2
            output11(i,:)=[0 1 0];
        case 3
            output11(i,:)=[0 0 1];
    end
end

for i=1:m2
    switch output2(i)
        case 1
            output22(i,:)=[1 0 0];
        case 2
            output22(i,:)=[0 1 0];
        case 3
            output22(i,:)=[0 0 1];
    end
end


%训练样本
input_train=data_train';
output_train=output11';
%预测样本
input_test=data_test';
output_test=output22';

%输入数据归一化
[inputn,inputps]=mapminmax(input_train);

%% 网络结构初始化
innum=118;
midnum=119;
outnum=3;
 

%权值初始化
w1=rands(midnum,innum);
b1=rands(midnum,1);
w2=rands(midnum,outnum);
b2=rands(outnum,1);

w2_1=w2;w2_2=w2_1;
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

%学习率
xite=0.1
alfa=0.01;

%% 网络训练
for ii=1:10
    E(ii)=0;
    for i=1:1:m1
       %% 网络预测输出 
        x=inputn(:,i);
        % 隐含层输出
        for j=1:1:midnum
            I(j)=inputn(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        % 输出层输出
        yn=w2'*Iout'+b2;
        
       %% 权值阀值修正
        %计算误差
        e=output_train(:,i)-yn;     
        E(ii)=E(ii)+sum(abs(e));
        
        %计算权值变化率
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j)));
            FI(j)=S*(1-S);
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3));
            end
        end
           
        w1=w1_1+xite*dw1';
        b1=b1_1+xite*db1';
        w2=w2_1+xite*dw2';
        b2=b2_1+xite*db2';
        
        w1_2=w1_1;w1_1=w1;
        w2_2=w2_1;w2_1=w2;
        b1_2=b1_1;b1_1=b1;
        b2_2=b2_1;b2_1=b2;
    end
end
 

%% 分类
inputn_test=mapminmax('apply',input_test,inputps);

for ii=1:1
    for i=1:m2
        %隐含层输出
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end



%% 结果分析
%根据网络输出找出数据属于哪类
for i=1:m2
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP网络预测误差
error=output_fore-output22';



%画出预测语音种类和实际语音种类的分类图
figure(1)
plot(output_fore,'r')
hold on
plot(output11','b')
legend('预测类别','实际类别')

%画出误差图
figure(2)
plot(error)
title('BP网络分类误差','fontsize',12)
xlabel('样本','fontsize',12)
ylabel('分类误差','fontsize',12)

%print -dtiff -r600 1-4

k=zeros(1,3);  
%找出判断错误的分类属于哪一类
for i=1:m2
    if error(i)~=0
        [b,c]=max(output_test(:,i));
        switch c
            case 1 
                k(1)=k(1)+1;
            case 2 
                k(2)=k(2)+1;
            case 3 
                k(3)=k(3)+1;
        end
    end
end

%找出每类的个体和
kk=zeros(1,3);
for i=1:m2
    [b,c]=max(output_test(:,i));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
        case 3
            kk(3)=kk(3)+1;
    end
end

%正确率
rightridio=(kk-k)./kk
