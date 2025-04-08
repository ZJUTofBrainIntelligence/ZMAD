clear;

% ==== 参数设置 ====
subIdx = 34;
sessIdx = 2;
motionIdx = 25; % 第几个动作

% ==== 路径配置 ====
basePath = 'D:\DataSet';
sEMGPath = fullfile(basePath, ['Sub_' num2str(subIdx)], ['Session_' num2str(sessIdx)], 'sEMG');

emgMatPath = fullfile(sEMGPath, 'emg.mat');
startPointPath = fullfile(sEMGPath, [num2str(sessIdx) 'sEMGStartPoint.mat']);
endPointPath   = fullfile(sEMGPath, [num2str(sessIdx) 'sEMGEndPoint.mat']);

% ==== 加载数据 ====
load(emgMatPath, 'emgArray');  % emgArray: N x 10，前两列时间戳，后8列为通道数据
SPoint = load(startPointPath, 'ESPoint').ESPoint;
EPoint = load(endPointPath, 'EEPoint').EEPoint;

% ==== 提取当前动作 ====
if motionIdx > length(SPoint) || motionIdx > length(EPoint)
    error('动作编号超过数据范围');
end

segment = emgArray(SPoint(motionIdx):EPoint(motionIdx), 3:10);  % 提取第3~10列（8通道数据）
numPoints = size(segment, 1);
t = linspace(0, numPoints, numPoints);  % 生成时间轴

% ==== StackedPlot 可视化 ====
figure;
stackedplot(segment);
title(sprintf('Sub %d - Session %d - Motion %d: 8-Channel sEMG', subIdx, sessIdx, motionIdx));

% ==== 三维绘图（8通道）====
figure('Position', [100, 100, 750, 416]);  % 设置图形窗口大小
hold on;
colors = lines(8);  % 生成8种不同的颜色
legendInfo = cell(1, 8);

for ch = 1:8
    plot3(t, ones(numPoints,1) * ch, segment(:, ch), 'Color', colors(ch, :));
    legendInfo{ch} = ['Channel ' num2str(ch)];
end

legend(legendInfo, 'Location', 'northeastoutside');
xlabel('Time / ms');
ylabel('Channels');
zlabel('Amplitude');
grid on;
view(-45, 45);
campos([-2000 -30 2000]);  % 调整相机视角
title(sprintf('3D sEMG Plot: Sub %d - Session %d - Motion %d', subIdx, sessIdx, motionIdx));
hold off;
