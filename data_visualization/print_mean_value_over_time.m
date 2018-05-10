clear all
clc
%% Plot mean values over time of first three channels (train) 

d = dlmread('../data_bci/sp1s_aa_train.txt');

output = d(:,1);
d = d(:,2:end);

data_channels = {};
number_channels = 28;
n_times = 50;

data_channels_right = {}
data_channels_left = {}
for i = 1:number_channels
   data_channels{end+1} = d(:,(i-1)*n_times+1:i*n_times);
   data_channels_right{end+1} = data_channels{i}(output == 0,:);
   data_channels_left{end+1} = data_channels{i}(output == 1,:);
end


% plot first three channels means
figure(5)
for i = 4:6
   subplot(1,3,i-3)
   plot(mean(data_channels_right{i},1),'r','Linewidth',1)
   hold on
   plot(mean(data_channels_left{i},1),'b','Linewidth',1)
end


%% Plot mean values over time of first three channels (test) 

d = dlmread('../data_bci/sp1s_aa_test.txt');

output = dlmread('../data_bci/labels_data_set_iv.txt');

data_channels = {};
number_channels = 28;
n_times = 50;

data_channels_right = {}
data_channels_left = {}
for i = 1:number_channels
   data_channels{end+1} = d(:,(i-1)*n_times+1:i*n_times);
   data_channels_right{end+1} = data_channels{i}(output == 0,:);
   data_channels_left{end+1} = data_channels{i}(output == 1,:);
end

% plot first three channels means
figure(1)
for i = 4:6
   subplot(1,3,i-3)
   plot(mean(data_channels_right{i},1),'r','Linewidth',1)
   hold on
   plot(mean(data_channels_left{i},1),'b','Linewidth',1)
end

