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
        if isa(samples_,'string')
            grouping_res = strings;
        else
            grouping_res = [];
        end
        sample_nums = [];
        grouped_impurity = [];% ����������������ز�����
        grouped_P = [];
        % ���features_(i)Ϊ1 ˵���÷�֧�ϸñ�ǩ��δ���ڷ���
        if features_(i) == 1
            % ����
            for j = 1:m
                pos = grouping_res == samples_(j,i);
                if sum(pos)
                    % �������� ����ͬһ��ǩ������������
                    sample_nums(pos) = sample_nums(pos) + 1;
                else   
                    % ����ǩ�������ӵ�ͳ�ƽ��
                    sample_nums = [sample_nums 1];
                    grouping_res(count) = samples_(j,i);
                    count = count + 1;
                end
            end
            % ����÷������Ĳ����ȼ�����
            % ����������ز�����
%             class = unique(samples_(:,n));
            for k = 1:size(grouping_res,1)
                sub_sample = samples_(samples_(:,i)==grouping_res{k},:);
                grouped_impurity = [grouped_impurity calculateImpurity(sub_sample)];
                grouped_P = [grouped_P sum(sub_sample(:,i)==grouping_res{k})/size(samples_(:,i),1)];
            end
            delta_impurity(i) = impurity_ - sum(grouped_P.*grouped_impurity);
        end
    end
    % ���ص�label����������
    temp = delta_impurity==max(delta_impurity);
    % ������ڶ�����һ���ı�ǩ ��ʹ�õ�һ��
    label = find(temp,1);
end
