% 计算熵不纯度
function res = calculateImpurity(examples_)
    P1 = 0;
    P2 = 0;
    P3 = 0;
    [m_,n_] = size(examples_);
    P1 = sum(examples_(:,n_) == 'setosa' );
    P2 = sum(examples_(:,n_) == 'versicolor');
    P3 = sum(examples_(:,n_) == 'virginica');
    P1 = P1 / m_;
    P2 = P2 / m_;
    P3 = P3 / m_;
    if P1 == 1 || P2 == 1 || P3 == 1 
        res = 0;
    elseif P1 == 0
        res = -(P2*log2(P2)+P3*log2(P3));
        elseif P2 == 0
        res = -(P1*log2(P1)+P3*log2(P3));
        elseif P3 == 0
        res = -(P1*log2(P1)+P2*log2(P2));
    else
        res = -(P1*log2(P1)+P2*log2(P2)+P3*log2(P3));
    end
end

%{
% 决策过程 获取信息增量最大的分类标准
function label = getBestlabel(impurity_,features_,samples_)
    % impurity_:划分前的熵不纯度
    % features_:当前可供分类的标签 是01矩阵
    % samples_:当前需要分类的样本
    [m,n]=size(samples_);
    delta_impurity = zeros(1,n-1);
    
    % 遍历每个特征 每个特征把m个样本分为t组 每组m_t个样本 计算每个特征的不纯度减少量delta_impurity(i)
    % 输入样本为m行n列矩阵 特征总数量为n-1
    
    for i = 1:n-1
        % 存放分类结果
        count = 1;
        grouping_res = strings;
        sample_nums = [];
        grouped_impurity = [];% 分类结果按分组计算熵不纯度
        grouped_P = [];
        % 如果features_(i)为1 说明该分支上该特征还未用于分类
        if features_(i) == 1
            % 分组
            for j = 1:m
                pos = grouping_res == samples_(j,i);
                if sum(pos)
                    % 分类样本 计算同一特征类别的样本数量
                    sample_nums(pos) = sample_nums(pos) + 1;
                else   
                    % 将特征的类别添加到统计结果
                    sample_nums = [sample_nums 1];
                    grouping_res(count) = samples_(j,i);
                    count = count + 1;
                    end
            end
            % 计算该分类结果的不纯度减少量
            % 按分组计算熵不纯度
            for k = grouping_res
                sub_sample = samples_(samples_(:,i)==k,:);
                grouped_impurity = [grouped_impurity calculateImpurity(sub_sample)];
                grouped_P = [grouped_P sum(sub_sample(:,n)=='是')/sum(samples_(:,i)==k)];
            end
            delta_impurity(i) = impurity_ - sum(grouped_P.*grouped_impurity);
        end
    end
    % 返回的label是索引数组
    temp = delta_impurity==max(delta_impurity);
    % 如果存在多个结果一样的标签 则使用第一个
    label = find(temp,1);
end
%}
