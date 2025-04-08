clear;

% ==== 参数设置 ====
subIdx = 1;
sessIdx = 1;
motionIdx = 32;

% ==== 路径配置 ====
basePath = 'D:\DataSet';
optitrackPath = fullfile(basePath, ['Sub_' num2str(subIdx)], ['Session_' num2str(sessIdx)], 'Optitrack');

matFilePath = fullfile(optitrackPath, 'RigidBody1.mat');
startFile = fullfile(optitrackPath, [num2str(sessIdx) 'StartPoint.mat']);
endFile   = fullfile(optitrackPath, [num2str(sessIdx) 'EndPoint.mat']);

% ==== 加载数据 ====
load(matFilePath, 'data');  % 变量名应为 data（来自RigidBody1.csv）
SPoint = load(startFile, 'SPoint').SPoint;
EPoint = load(endFile, 'EPoint').EPoint;

% ==== 数据提取 ====
X = data(:, 3);  % X, Y, Z 是 data 的第3-5列
Y = data(:, 4);
Z = data(:, 5);

% 检查动作编号是否超出范围
if motionIdx > length(SPoint)
    error('动作编号 %d 超出范围 (共 %d 个)', motionIdx, length(SPoint));
end

rangeX = X(SPoint(motionIdx):EPoint(motionIdx));
rangeY = Y(SPoint(motionIdx):EPoint(motionIdx));
rangeZ = Z(SPoint(motionIdx):EPoint(motionIdx));

% ==== 绘图 ====
figure('Position', [680, 558, 560, 420]);
hold on;
plot3(rangeX, rangeY, rangeZ, 'LineWidth', 2);

xlabel('X Axis');
ylabel('Y Axis');
zlabel('Z Axis');
grid on;

view(-158, 5);  % 视角调整
campos([-1.81, 1.60, 1.21]);  % 相机位置
title(sprintf('Sub %d - Session %d - Motion %d: 3D Motion Trajectory', subIdx, sessIdx, motionIdx));

hold off;
