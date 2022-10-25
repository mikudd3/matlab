% ���ɾ�����
function node = makeTree(features,examples,deepth,branch)
    % feature:�����������ݵ���������
    % examples:����
    % deepth:������ȣ�ÿ������һ��������ǩ��Ӧ��ֵ����

    % value:����������Ϊnull���ʾ�ýڵ��Ƿ�֧�ڵ�
    % label:�ڵ㻮�ֱ�ǩ
    % branch:��ֵ֧
    % children:�ӽڵ�
    node = struct('value','null','label',[],'branch',branch,'children',[]);
    
    [m,n] = size(examples);
    sample = examples(1,n);
    check_res = true;
    for i = 1:m
        if sample ~= examples(i,n)
            check_res = false;
        end
    end
    % ��������ȫΪͬһ������ ����ΪҶ�ڵ�
    if check_res
        node.value = examples(1,n);
        return;
    end
    
    % ������
    impurity = calculateImpurity(examples);
    % ѡ����ʵ�����
    bestLabel = getBestlabel(impurity,deepth,examples);
    deepth(bestLabel) = 0;
    node.label = features(bestLabel);
    
    % ����
    if isa(examples,'string')
            grouping_res = strings;
        else
            grouping_res = [];
    end
    count = 1;
    for i = 1:m
        pos = grouping_res == examples(i,bestLabel);
        if sum(pos)
            % �������� ����ͬһ��ǩ������������
        else   
            % ����ǩ�������ӵ�ͳ�ƽ��
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
