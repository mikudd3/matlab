%%清空环境变量
clc
clear
close all

%%
%导入数据

% data_train = readmatrix("Doorbell_mixed_train_IM.xlsx");
% data_test = readmatrix("Doorbell_mixed_test_IM.xlsx");
% [m1,n1] = size(data_train);
% [m2,n2] = size(data_test);
% 
% %%随机选取数据
% train_index = randperm(m1,10000);
% Train_sample = data_train(train_index,:);
% test_index = randperm(m2,2000);
% Test_sample = data_test(test_index,:);
% writematrix(Train_sample,'train.xlsx');
% writematrix(Test_sample,'test.xlsx');

data_train = readmatrix("train.xlsx");
train_label = readmatrix("train__label_9.xlsx");
[m1,n1] = size(data_train);

data_test = readmatrix("test.xlsx");
test_label = readmatrix('test_label_9.xlsx');
[m2,n2] = size(data_test);


%%数据归一化
[p_train,ps_input] = mapminmax(data_train',0,1);
p_test = mapminmax('apply',data_test',ps_input);

t_train = ind2vec(train_label');
t_test = ind2vec(test_label');

%%创建网络
rbf_spread = 100;
net = newrbe(p_train,t_train,rbf_spread);

%%仿真测试
t_sim = sim(net,p_test);

%%数据反归一化
pre = vec2ind(t_sim);
pre = pre';

%%性能评价
%准确率
accuracy = sum((test_label==pre))/numel(test_label);

%%数据排序
[test_label,index] = sort(test_label);
pre = pre(index);


%%画图
% figure 
% plot(1:m2,test_label,'r-*',1:m2,pre,'b-o','LineJoin',1);
% legend('真实值','预测值');
% xlabel('预测样本');
% ylabel('预测结果');
% string = {['准确率=' num2str(accuracy)] '%'};
% title(string);
% xlim([1,m2]);
% grid
plot(test_label,'bo')
hold on
plot(pre,'r*')
grid on
title('accuracy =', accuracy(1,1))
xlabel('样本序号')
ylabel('类型')
legend("实际类型","预测类型")
set(gca,'fontsize',12)



%%混淆矩阵
figure
cm = confusionchart(test_label,pre);
cm.Title = '混淆矩阵';
cm.ColumnSummary = 'column - normalized';
cm.RowSummary = 'row-normalized';

[A,~] = confusionmat(test_label,pre);



