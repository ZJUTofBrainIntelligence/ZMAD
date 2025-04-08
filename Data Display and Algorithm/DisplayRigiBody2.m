clear;

% 设置路径
Path = 'D:\DataSet\Sub_1\Session_1\Optitrack\';
i = 1; % 当前动作编号

% 文件路径
fileName1 = fullfile(Path, '1StartPoint.mat');
fileName2 = fullfile(Path, '1EndPoint.mat');
matFilePath = fullfile(Path, 'RigidBody1.mat');

% 加载动捕数据（已转换为 .mat 格式，变量名为 data）
load(matFilePath, 'data');

% 提取 X/Y/Z（第3~5列），并转置为行向量
X_data = data(:,3)';
Y_data = data(:,4)';
Z_data = data(:,5)';

% 加载起止点
SPoint = load(fileName1, 'SPoint').SPoint;
EPoint = load(fileName2, 'EPoint').EPoint;

% 检查索引合法性
if i > length(SPoint) || i > length(EPoint)
    error('动作编号 i 超出起止点索引范围！');
end

% 绘图 - X
figure;
plot(X_data(SPoint(i):EPoint(i)));
title(sprintf('Sub 34 Session 1 Motion %d - X', i));
legend('X');

% 绘图 - Y
figure;
plot(Y_data(SPoint(i):EPoint(i)));
title(sprintf('Sub 34 Session 1 Motion %d - Y', i));
legend('Y');

% 绘图 - Z
figure;
plot(Z_data(SPoint(i):EPoint(i)));
title(sprintf('Sub 34 Session 1 Motion %d - Z', i));
legend('Z');

clear;
