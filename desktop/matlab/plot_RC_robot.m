%% plot

close all

lwidth = 3;

figure(1);

plot(time_axis(1:length(target_Out)) - start_train, arm_x(2, 1:1450), 'linewidth', lwidth);

xlim([time_axis([1 end]) - start_train]);
ylabel('Joint angle [rad]');
xlabel('time (ms)');


figure(2)

plot(time_axis(1:length(target_Out)) - start_train, squeeze(Out_test_history(2, :, 1)), 'linewidth', lwidth);

xlim([time_axis([1 end]) - start_train]);
ylim([-0.1 1.4]);
ylabel('RC output');
xlabel('time (ms)');


figure(3)

plot(time_axis(1:length(target_Out)) - start_train, u_history(2, :, 1), 'linewidth', lwidth);

xlim([time_axis([1 end]) - start_train]);
ylabel('Torque');
xlabel('time (ms)');