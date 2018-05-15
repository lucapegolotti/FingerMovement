function [ ] = plot_mean( m_train_r,m_train_l,vec_ch, n_subplot, current_subplot, n_times,moreplots,normal )
colorline_r = 'rk';
colorline_l = 'bm';
typeline='--';
typemarker='o*';

n_channels_plot = numel(vec_ch);

if n_times == 500
    t = 1:500;
else
    t = 7:10:500;
end
count=0;
for i = 1:n_channels_plot
    count = count+1;
    subplot(n_subplot,n_channels_plot,current_subplot*n_channels_plot+count)
    if normal
        plot(t,m_train_r{vec_ch(i)},'r','Linewidth',1)
    else
        plot(t,m_train_r{vec_ch(i)},strcat(colorline_r(moreplots),typeline(moreplots),typemarker(moreplots)),'Linewidth',1)
    end
    hold on
    
    if normal
        plot(t,m_train_l{vec_ch(i)},'b','Linewidth',1)
    else
        plot(t,m_train_l{vec_ch(i)},strcat(colorline_l(moreplots),typeline(moreplots),typemarker(moreplots)),'Linewidth',1)
    end
    xlim([0 500])
    if current_subplot == 0
        title(strcat('Channel',{' '},num2str(vec_ch(i))));
        if i == 1
       ylabel('Train Dataset','FontWeight','bold')
        end
    else
        
        %title(strcat('Test - channel',{' '},num2str(vec_ch(i))));
        if i == 1
       ylabel('Test Dataset','FontWeight','bold')
        end
    end
    set(gca,'fontsize',18)
end

end

