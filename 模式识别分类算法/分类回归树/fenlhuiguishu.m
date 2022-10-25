clc
clear

%%
%导入数据训练数据和测试数据
train_data = csvread('Doorbell_mixed_train_IM.csv',1,0);
test_data = csvread('Doorbell_mixed_test_IM.csv',1,0);
train_index = 2:size(train_data)+1;
test_index = 2:size(test_data)+1;

%导入训练标签（三类）
fid3_tr = fopen('label3_train_IM.csv');
label3_tr = textscan(fid3_tr,'%s');
fclose(fid3_tr);
tr_label3 = string(label3_tr{1,1}(train_index));

%导入测试标签（三类）
fid3_te = fopen('label3_test_IM.csv');
label3_te = textscan(fid3_te,'%s');
fclose(fid3_te);
te_label3 = string(label3_te{1,1}(test_index));

%导入训练标签（九类）
fid9_tr = fopen('label9_train_IM.csv');
label9_tr = textscan(fid9_tr,'%s');
fclose(fid9_tr);
tr_label9 = string(label9_tr{1,1}(train_index));

%导入测试标签（九类）
fid9_te = fopen('label9_test_IM.csv');
label9_te = textscan(fid9_te,'%s');
fclose(fid9_te);
te_label9 = string(label9_te{1,1}(test_index));

train_set = train_data(:,1:end-3);
test_set = test_data(:,1:end-3);


%%
%分三类
ctree = fitctree(train_set,tr_label3); 
[pre_label1, scores] = predict(ctree,test_set);
view(ctree,'mode','graph');
accuracy1 = sum(strcmp(te_label3,string(pre_label1)))/numel(te_label3)
confusionchart(te_label3,string(pre_label1))

class9 = unique(te_label3);
 for i =1:size(pre_label1,1)
     for j= 1:size(class9,1)
         if class9(j) == pre_label1(i)
             te3(i)= j;
         end
         if class9(j) == pre_label1(i)
             pre3(i)= j;
         end
     end
 end
[te3,index] = sort(te3);
pre3 = pre3(index);
plot(te3, 'bo') 
hold on
plot(pre3,'r*')
grid on
title( 'accuracy =', accuracy1 )
xlabel('样本序号')
ylabel( '类型')
legend("实际类型", "预测类型")
set(gca, 'fontsize',12)

%%
%分九类
ctree1 = fitctree(train_set,tr_label9); 
[pre_label2, scores] = predict(ctree1,test_set);
view(ctree1,'mode','graph');
accuracy2 = sum(strcmp(te_label9,string(pre_label2)))/numel(te_label9)
confusionchart(te_label9,string(pre_label2))
class9 = unique(te_label9);
 for i =1:size(pre_label2,1)
     for j= 1:size(class9,1)
         if class9(j) == pre_label2(i)
             te(i)= j;
         end
         if class9(j) == pre_label2(i)
             pre(i)= j;
         end
     end
 end
[te,index] = sort(te);
pre = pre(index);
plot(te, 'bo') 
hold on
plot(pre,'r*')
grid on
title( 'accuracy =', accuracy2 )
xlabel('样本序号')
ylabel( '类型')
legend("实际类型", "预测类型")
set(gca, 'fontsize',12)