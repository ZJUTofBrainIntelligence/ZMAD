clear;

% ==== 参数设置 ====
subIdx = 1;
sessIdx = 1;
motionIdx = 1;

% ==== 路径配置 ====
basePath = 'D:\DataSet';
accPath = fullfile(basePath, ['Sub_' num2str(subIdx)], ['Session_' num2str(sessIdx)], 'Acceleration');

matFile = fullfile(accPath, 'acceleration.mat');
startFile = fullfile(accPath, [num2str(sessIdx) 'AccStartPoint.mat']);
endFile   = fullfile(accPath, [num2str(sessIdx) 'AccEndPoint.mat']);

% ==== 加载数据 ====
load(matFile, 'Acceleration');  % 变量为 Acceleration，列为 X Y Z
SPoint = load(startFile, 'AccSPoint').AccSPoint;
EPoint = load(endFile, 'AccEPoint').AccEPoint;

% ==== 提取数据 ====
accX = Acceleration(:, 3);
accY = Acceleration(:, 4);
accZ = Acceleration(:, 5);

% 范围检查
if motionIdx > length(SPoint)
    error('动作编号 %d 超出范围 (共 %d 个)', motionIdx, length(SPoint));
end

rangeX = accX(SPoint(motionIdx):EPoint(motionIdx));
rangeY = accY(SPoint(motionIdx):EPoint(motionIdx));
rangeZ = accZ(SPoint(motionIdx):EPoint(motionIdx));

M = [rangeX, rangeY, rangeZ];
stackedplot(M);

% ==== z轴映射函数 ====
z_transform = @(z) (z > -0.5) .* (z - 0.5) * 0.5 + (z > -0.5) * 0.5 + (z <= -0.5) .* z;

zX = z_transform(rangeX);
zY = z_transform(rangeY);
zZ = z_transform(rangeZ);

numPoints = EPoint(motionIdx) - SPoint(motionIdx) + 1;
t = linspace(0, numPoints, numPoints);

% ==== 绘制 3D 多通道加速度数据 ====
figure('Position', [680, 500, 650, 420]);
hold on;
colors = lines(3);

plot3(t, ones(numPoints,1)*1, zX, 'Color', colors(1,:), 'LineWidth', 1.2);
plot3(t, ones(numPoints,1)*2, zY, 'Color', colors(2,:), 'LineWidth', 1.2);
plot3(t, ones(numPoints,1)*3, zZ, 'Color', colors(3,:), 'LineWidth', 1.2);

grid on;
xlabel('Time/ms');
ylabel('Channels');
zlabel('Amplitude');

zticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.6, 1.2]);
zlim([-1.2, 1.2]);

view(-48, 27);
campos([-1052.3, -8.6, 9.436]);

title(sprintf('Sub %d - Session %d - Motion %d: Accelerometer Data', subIdx, sessIdx, motionIdx));
hold off;
