% �����ز�����
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
% ���߹��� ��ȡ��Ϣ�������ķ����׼
function label = getBestlabel(impurity_,features_,samples_)
    % impurity_:����ǰ���ز�����
    % features_:��ǰ�ɹ�����ı�ǩ ��01����
    % samples_:��ǰ��Ҫ���������
    [m,n]=size(samples_);
    delta_impurity = zeros(1,n-1);
    
    % ����ÿ������ ÿ��������m��������Ϊt�� ÿ��m_t������ ����ÿ�������Ĳ����ȼ�����delta_impurity(i)
    % ��������Ϊm��n�о��� ����������Ϊn-1
    
    for i = 1:n-1
        % ��ŷ�����
        count = 1;
        grouping_res = strings;
        sample_nums = [];
        grouped_impurity = [];% ����������������ز�����
        grouped_P = [];
        % ���features_(i)Ϊ1 ˵���÷�֧�ϸ�������δ���ڷ���
        if features_(i) == 1
            % ����
            for j = 1:m
                pos = grouping_res == samples_(j,i);
                if sum(pos)
                    % �������� ����ͬһ����������������
                    sample_nums(pos) = sample_nums(pos) + 1;
                else   
                    % �������������ӵ�ͳ�ƽ��
                    sample_nums = [sample_nums 1];
                    grouping_res(count) = samples_(j,i);
                    count = count + 1;
                    end
            end
            % ����÷������Ĳ����ȼ�����
            % ����������ز�����
            for k = grouping_res
                sub_sample = samples_(samples_(:,i)==k,:);
                grouped_impurity = [grouped_impurity calculateImpurity(sub_sample)];
                grouped_P = [grouped_P sum(sub_sample(:,n)=='��')/sum(samples_(:,i)==k)];
            end
            delta_impurity(i) = impurity_ - sum(grouped_P.*grouped_impurity);
        end
    end
    % ���ص�label����������
    temp = delta_impurity==max(delta_impurity);
    % ������ڶ�����һ���ı�ǩ ��ʹ�õ�һ��
    label = find(temp,1);
end
%}
