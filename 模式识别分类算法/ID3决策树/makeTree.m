% 生成决策树
function node = makeTree(features,examples,deepth,branch)
    % feature:样本分类依据的所有特征
    % examples:样本
    % deepth:树的深度，每被分类一次与分类标签对应的值置零

    % value:分类结果，若为null则表示该节点是分支节点
    % label:节点划分标签
    % branch:分支值
    % children:子节点
    node = struct('value','null','label',[],'branch',branch,'children',[]);
    
    [m,n] = size(examples);
    sample = examples(1,n);
    check_res = true;
    for i = 1:m
        if sample ~= examples(i,n)
            check_res = false;
        end
    end
    % 若样本中全为同一分类结果 则作为叶节点
    if check_res
        node.value = examples(1,n);
        return;
    end
    
    % 计算熵
    impurity = calculateImpurity(examples);
    % 选择合适的特征
    bestLabel = getBestlabel(impurity,deepth,examples);
    deepth(bestLabel) = 0;
    node.label = features(bestLabel);
    
    % 分类
    if isa(examples,'string')
            grouping_res = strings;
        else
            grouping_res = [];
    end
    count = 1;
    for i = 1:m
        pos = grouping_res == examples(i,bestLabel);
        if sum(pos)
            % 分类样本 计算同一标签类别的样本数量
        else   
            % 将标签的类别添加到统计结果
            grouping_res(count) = examples(i,bestLabel);
            count = count + 1;
        end
    end
    
    for k = grouping_res
        if sum(deepth(:)) == 0 
            break;
        end
        sub_sample = examples(examples(:,bestLabel)==k,:);
        node.children = [node.children makeTree(features,sub_sample,deepth,k)];
    end
    
end
