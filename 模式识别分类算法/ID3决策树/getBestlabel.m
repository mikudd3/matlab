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
        if isa(samples_,'string')
            grouping_res = strings;
        else
            grouping_res = [];
        end
        sample_nums = [];
        grouped_impurity = [];% 分类结果按分组计算熵不纯度
        grouped_P = [];
        % 如果features_(i)为1 说明该分支上该标签还未用于分类
        if features_(i) == 1
            % 分组
            for j = 1:m
                pos = grouping_res == samples_(j,i);
                if sum(pos)
                    % 分类样本 计算同一标签类别的样本数量
                    sample_nums(pos) = sample_nums(pos) + 1;
                else   
                    % 将标签的类别添加到统计结果
                    sample_nums = [sample_nums 1];
                    grouping_res(count) = samples_(j,i);
                    count = count + 1;
                end
            end
            % 计算该分类结果的不纯度减少量
            % 按分组计算熵不纯度
%             class = unique(samples_(:,n));
            for k = 1:size(grouping_res,1)
                sub_sample = samples_(samples_(:,i)==grouping_res{k},:);
                grouped_impurity = [grouped_impurity calculateImpurity(sub_sample)];
                grouped_P = [grouped_P sum(sub_sample(:,i)==grouping_res{k})/size(samples_(:,i),1)];
            end
            delta_impurity(i) = impurity_ - sum(grouped_P.*grouped_impurity);
        end
    end
    % 返回的label是索引数组
    temp = delta_impurity==max(delta_impurity);
    % 如果存在多个结果一样的标签 则使用第一个
    label = find(temp,1);
end
