function SE = sampleE(data, m, r)
% % SampEn  计算时间序列data的样本熵
% % 输入：data是数据一维行向量
% %      m重构维数，一般选择1或2，优先选择2，一般不取m>2
% %      r 阈值大小，一般选择r=0.1~0.25*Std(data)
% % 输出：SampEnVal样本熵值大小
data = data(:)';          %将矩阵 data 中的元素按列顺序排列成一个行向量
N = length(data);         %获取数据总数
Bmi = 0;                  %Bmi用于记录Xm(i)和Xm(j)的距离小于r的个数
Ami = 0;                  %Ami用于记录Xm+1(i)和Xm+1(j)的距离小于r的个数
%r = r * std(data);
% 分段计算距离，x1为长度为m的序列，x2为长度为m+1的序列
for k = N-m : -1 : 1
    x1(k, :) = data(k:k + m - 1);  
    x2(k, :) = data(k:k + m); 
end
for k = N - m:-1:1
    x1temp1 = x1(k,:);    %获取长度为m的其中一个序列
    x1temp2 = ones(N - m, 1)*x1temp1; %将这个序列复制成N-m的矩阵方便后续使用x1temp1跟其他序列求距离
    dx1(k, :) = max(abs(x1temp2 - x1), [], 2)';        % abs(x1temp2 - x1)：求x1temp1跟所有其他长度为m的序列的距离(包括x1temp1自己，所以后续sum求数量要减1)
                                                       % 如果 A 为矩阵，则max(A,[],2)是包含每一行的最大值的列向量(就是求出距离的最大值)
    Bmi = Bmi  + (sum(dx1(k, :) < r) - 1)/(N - m - 1); % 求Xm(i)和Xm(j)的距离小于r的个数，由于之前矩阵相减中包含了自己减自己，所以总数小于r的要减1
    
    %x2序列计算，和x1同样方法
    x2temp1 = x2(k,:);    %获取长度为m+1的其中一个序列
    x2temp2 = ones(N - m, 1)*x2temp1; 
    dx2(k, :) = max(abs(x2temp2 - x2), [], 2)';       
    Ami = Ami  + (sum(dx2(k, :) < r) - 1)/(N - m - 1);
end

Bmr = Bmi/(N - m);
Amr = Ami/(N - m);
SE = log(Bmr/Amr);


