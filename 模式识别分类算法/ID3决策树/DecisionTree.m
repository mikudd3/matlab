clc
clear
load fisheriris
d = [1,4,7,10];
for i = 1:4
    Xmax = max(meas(:,i));
    Xmin = min(meas(:,i));
    t = (Xmax - Xmin)/3 ;
    for k = 1:150
        if meas(k,i) < (Xmin + t) || meas(k,i) == (Xmin + t)
            meas(k,i) = d(i)+1;
        elseif meas(k,i) < (Xmin + 2*t)|| meas(k,i) == (Xmin + 2*t)
            meas(k,i) = d(i)+2;
        elseif meas(k,i) > (Xmin + 2*t)
            meas(k,i) = d(i)+3;
        end
    end   
end
train_index = randperm(150,50);
sample_number = 1:size(meas);
sample_number(train_index) = [];
test_index = sample_number;
train_set = meas(train_index,:);
test_set = meas(test_index,:);
tr_label = string(species(train_index));
te_label = string(species(test_index));
feature_name = ["sepals length","sepals width","petals length","petals width","category"];
train = [train_set tr_label];
deepth = ones(1,size(train,2)-1);
% 生成树
rootNode = makeTree(feature_name,train,deepth,'null');
% 画出决策树
drawTree(rootNode);
%}