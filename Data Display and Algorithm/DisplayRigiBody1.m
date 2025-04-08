clc;
clear;

% 设置目标路径
basePath = 'D:\DataSet\Sub_1\Session_1\Optitrack';

% 文件路径设置
matFilePath = fullfile(basePath, 'RigidBody1.mat');
startPointFile = fullfile(basePath, '1StartPoint.mat');
endPointFile   = fullfile(basePath, '1EndPoint.mat');

% 加载动捕数据
load(matFilePath, 'data');  % data 是 Nx5，取第3~5列是 X, Y, Z

X_data = data(:,3)';
Y_data = data(:,4)';
Z_data = data(:,5)';

% 加载起止点
SPointData = load(startPointFile); % SPoint
EPointData = load(endPointFile);   % EPoint
SPoint = SPointData.SPoint;
EPoint = EPointData.EPoint;

%% 绘图
figure;
hold on;
for i = 1 : 37
    plot(X_data(SPoint(i):EPoint(i)));
end
hold off;
legend('X');

figure;
hold on;
for i = 1 : 37
    plot(Y_data(SPoint(i):EPoint(i)));
end
hold off;
legend('Y');

figure;
hold on;
for i = 1 : 37
    plot(Z_data(SPoint(i):EPoint(i)));
end
hold off;
legend('Z');

clear;
