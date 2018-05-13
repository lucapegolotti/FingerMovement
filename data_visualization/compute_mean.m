function [ m_train_r,m_train_l,data_channels_right,data_channels_left ] = compute_mean( d,output,n_times )

data_channels = {};
number_channels = 28;

data_channels_right = {};
data_channels_left = {};
for i = 1:number_channels
    data_channels{end+1} = d(:,(i-1)*n_times+1:i*n_times);
    data_channels_right{end+1} = data_channels{i}(output == 0,:);
    data_channels_left{end+1} = data_channels{i}(output == 1,:);
end


m_train_r = {};
m_train_l = {};


for i = 1:number_channels
    m_train_r{end+1} = mean(data_channels_right{i},1);
    m_train_l{end+1} = mean(data_channels_left{i},1);
end

end

