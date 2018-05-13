clear
close all

clc

%% 1000 Hz -  Compute mean values over time of all 28 channels 
n_times1000 = 500;

% Train
d_train1000 = dlmread('../data_bci/sp1s_aa_train_1000Hz.txt');
output_train = d_train1000(:,1);
d_train1000 = d_train1000(:,2:end);
[ m_train_1000hz_first100_r, m_train_1000hz_first100_l ] = compute_mean( d_train1000(1:100,:), output_train(1:100),n_times1000 );
[ m_train_1000hz_r, m_train_1000hz_l ] = compute_mean( d_train1000,output_train,n_times1000 );

% Test 
d_test1000 = dlmread('../data_bci/sp1s_aa_test_1000Hz.txt');
output_test = dlmread('../data_bci/labels_data_set_iv.txt');
[ m_test_1000hz_r, m_test_1000hz_l ] = compute_mean( d_test1000,output_test,n_times1000 );


%% Downsampling -  Compute mean values over time of all 28 channels 
n_times = 50;

% Train - first 100
d_train = dlmread('../data_bci/sp1s_aa_train.txt');
d_train = d_train(:,2:end);
[ m_train_r, m_train_l, data_channels_train_right, data_channels_train_left ] = compute_mean( d_train,output_train,n_times );

% Test 
d_test = dlmread('../data_bci/sp1s_aa_test.txt');
[ m_test_r, m_test_l, data_channels_test_right, data_channels_test_left] = compute_mean( d_test,output_test,n_times );



%% Plots
vec_ch = [3,10,25];

figure;
plot_mean( m_train_1000hz_first100_r,m_train_1000hz_first100_l,vec_ch, 2, 0, n_times1000 ,1,1)
plot_mean( m_test_1000hz_r,m_test_1000hz_l,vec_ch, 2, 1, n_times1000,1,1 )
legend('Mean R','Mean L')

figure;
plot_mean( m_train_1000hz_r,m_train_1000hz_l,vec_ch, 2, 0, n_times1000,1,1 )
plot_mean( m_test_1000hz_r,m_test_1000hz_l,vec_ch, 2, 1, n_times1000,1,1 )
legend('Mean R','Mean L')

figure;
plot_mean( m_train_r,m_train_l,vec_ch, 2, 0, n_times,1,1 )
plot_mean( m_test_r,m_test_l,vec_ch, 2, 1, n_times,1,1 )
legend('Mean R','Mean L')

figure;
plot_mean( m_train_1000hz_r,m_train_1000hz_l,vec_ch, 2, 0, n_times1000,1,0 )
plot_mean( m_test_1000hz_r,m_test_1000hz_l,vec_ch, 2, 1, n_times1000,1,0 )
hold on
plot_mean( m_train_r,m_train_l,vec_ch, 2, 0, n_times,2,0)
plot_mean( m_test_r,m_test_l,vec_ch, 2, 1, n_times,2 ,0)
legend('Mean R 1000Hz','Mean L 1000Hz','Mean R Downsampled','Mean L Downsampled')

set(gca,'fontsize',18)





%% FIGURE 4 - Downsampled - Plot few random values over time of some channels + mean (train) 

figure
count=0;
t= 7:10:500;
for i = 1:numel(vec_ch)
    count = count+1;
   subplot(2,numel(vec_ch),count)
   plot(t,data_channels_train_right{vec_ch(i)}(10,:),'r','Linewidth',1)
   hold on
   plot(t,data_channels_train_right{vec_ch(i)}(35,:),'r--','Linewidth',1)
   plot(t,data_channels_train_right{vec_ch(i)}(78,:),'r-.','Linewidth',1)
   plot(t,data_channels_train_left{vec_ch(i)}(3,:),'b','Linewidth',1)
   plot(t,data_channels_train_left{vec_ch(i)}(56,:),'b--','Linewidth',1)
   plot(t,data_channels_train_left{vec_ch(i)}(120,:),'b-.','Linewidth',1)
   xlim([0 500])
   
   plot(t,m_train_r{vec_ch(i)},'k-.','Linewidth',2)
   plot(t,m_train_l{vec_ch(i)},'k-','Linewidth',2)
   xlim([0 500])
   title(strcat('Train datset - channel',{' '},num2str(vec_ch(i))))
   set(gca,'FontSize',18)
   
end

% Downsampled - Plot few random values over time of some channels + mean  (test) 
for i = 1:numel(vec_ch)
    count=count+1;
   subplot(2,numel(vec_ch),count)
   plot(t,data_channels_test_right{vec_ch(i)}(15,:),'r','Linewidth',1)
   hold on
   plot(t,data_channels_test_right{vec_ch(i)}(36,:),'r--','Linewidth',1)
   plot(t,data_channels_test_right{vec_ch(i)}(48,:),'r-.','Linewidth',1)
   plot(t,data_channels_test_left{vec_ch(i)}(3,:),'b','Linewidth',1)
   plot(t,data_channels_test_left{vec_ch(i)}(12,:),'b--','Linewidth',1)
   plot(t,data_channels_test_left{vec_ch(i)}(36,:),'b-.','Linewidth',1)
   xlim([0 500])
   
   plot(t,m_test_r{vec_ch(i)},'k-','Linewidth',2)
   plot(t,m_test_l{vec_ch(i)},'k-.','Linewidth',2)
   title(strcat('Test datset - channel',{' '},num2str(vec_ch(i))))
   set(gca,'FontSize',18)
end

legend('Sample 15 - R','Sample 36 - R','Sample 48 - R','Sample 3- L','Sample 12 - L','Sample 36 - L','Mean R on test datest','Mean L  on test datest')

subplot(2,3,1);ylim([-40 130])
subplot(2,3,4);ylim([-40 130])
subplot(2,3,2);ylim([-30 130])
subplot(2,3,5);ylim([-30 130])
subplot(2,3,3);ylim([-80 90])
subplot(2,3,6);ylim([-80 90])