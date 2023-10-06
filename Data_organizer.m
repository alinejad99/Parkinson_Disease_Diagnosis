clear; clc; close all;

Control_1 = load('Control/Control1021.mat');
Control_2 = load('Control/Control1041.mat');
Control_3 = load('Control/Control1061.mat');
Control_4 = load('Control/Control1081.mat');
Control_5 = load('Control/Control1101.mat');
Control_6 = load('Control/Control1111.mat');
Control_7 = load('Control/Control1191.mat');
Control_8 = load('Control/Control1201.mat');
Control_9 = load('Control/Control1211.mat');
Control_10 = load('Control/Control1231.mat');
Control_11 = load('Control/Control1291.mat');
Control_12 = load('Control/Control1351.mat');
Control_13 = load('Control/Control1381.mat');
Control_14 = load('Control/Control1411.mat');

numArrays = 14;
for i = 1:numArrays
    eval(sprintf('ctl_%d = [Control_%d.EEG.data];', i, i));
    eval(sprintf('clear Control_%d;', i));
end
%%
numArrays = 14;
% Delete the 64th channel from ctl_14
ctl_14 = ctl_14(1:63, :);

minSecondDim = inf; 
for i = 1:numArrays
    arraySize = size(eval(sprintf('ctl_%d', i)));
    minSecondDim = min(minSecondDim, arraySize(2));
end

for i = 1:numArrays
    array = eval(sprintf('ctl_%d', i));
    eval(sprintf('ctl_%d = array(:, 1:minSecondDim);', i));
end

%%
% Define the original sample rate and the desired sample rate
originalSampleRate = 500;
desiredSampleRate = 100;

% Calculate the downsampling factor
downsamplingFactor = originalSampleRate / desiredSampleRate;

% Downsample each array
for i = 1:numArrays
    % Get the current array
    array = eval(sprintf('ctl_%d', i));
    
    % Downsample the array
    downsampledArray = array(:, 1:downsamplingFactor:end);
    
    % Uctlate the array with the downsampled data
    eval(sprintf('ctl_%d = downsampledArray;', i));
end

%%

% Define the segment length
segmentLength = 1000;

% Calculate the number of segments
numSegments = floor(size(ctl_1, 2) / segmentLength);

% Adjust the size of the arrays if necessary
numElements = numSegments * segmentLength;
for i = 1:numArrays
    eval(sprintf('ctl_%d = ctl_%d(:, 1:numElements);', i, i));
end

% Get the actual number of segments
numSegments = size(ctl_1, 2) / segmentLength;

% Segment each array
for i = 1:numArrays
    % Get the current array
    array = eval(sprintf('ctl_%d', i));
    
    % Reshape the array to a 3D array
    array = reshape(array, 63, segmentLength, numSegments);
    
    % Uctlate the original array
    eval(sprintf('ctl_%d = array;', i));
end
%%
% Create a 4D array to store the concatenated data
Control = cat(3, ctl_1, ctl_2, ctl_3, ctl_4, ctl_5, ctl_6, ctl_7, ctl_8, ctl_9, ctl_10, ctl_11, ctl_12, ctl_13, ctl_14);
save('Control.mat', 'Control');
    
%
% for i = 1:numArrays
%     ctl = eval(sprintf('ctl_%d', i));
%     filename = sprintf('ctl_%d.mat', i);
%     save(filename, 'ctl');
% end
    
    
    
    
