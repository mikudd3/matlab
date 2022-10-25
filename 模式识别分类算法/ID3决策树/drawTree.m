% 画出决策树
function [] = drawTree(node)
    % 遍历树
    nodeVec = [];
    nodeSpec = [];
    edgeSpec = [];
    [nodeVec,nodeSpec,edgeSpec,total] = travesing(node,0,0,nodeVec,nodeSpec,edgeSpec);
    treeplot(nodeVec);
    [x,y] = treelayout(nodeVec);
    [m,n] = size(nodeVec);
    x = x';
    y = y';
    text(x(:,1),y(:,1),nodeSpec,'VerticalAlignment','bottom','HorizontalAlignment','right');
    x_branch = [];
    y_branch = [];
    for i = 2:n
        x_branch = [x_branch; (x(i,1)+x(nodeVec(i),1))/2];
        y_branch = [y_branch; (y(i,1)+y(nodeVec(i),1))/2];
    end
    text(x_branch(:,1),y_branch(:,1),edgeSpec(1,2:n),'VerticalAlignment','bottom','HorizontalAlignment','right');
end

% 遍历树
function [nodeVec,nodeSpec,edgeSpec,current_count] = travesing(node,current_count,last_node,nodeVec,nodeSpec,edgeSpec)
    nodeVec = [nodeVec last_node];
    if node.value == 'null'
        nodeSpec = [nodeSpec node.label];
    else
        if node.value == 'setosa'
            nodeSpec = [nodeSpec 'setosa'];
        elseif node.value == 'versicolor'
            nodeSpec = [nodeSpec 'versicolor'];
        else
            nodeSpec = [nodeSpec 'virginica'];
        end
    end
    edgeSpec = [edgeSpec node.branch];
    current_count = current_count + 1;
    current_node = current_count;
    if node.value ~= 'null'
        return;
    end
    for next_ndoe = node.children
        [nodeVec,nodeSpec,edgeSpec,current_count] = travesing(next_ndoe,current_count,current_node,nodeVec,nodeSpec,edgeSpec);
    end
end
