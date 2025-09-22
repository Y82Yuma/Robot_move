j = 1;
i = 1;

figure(4)
if strcmp(task, 'point')
    target = culc_kinematics([1; 1], L);
    plot(target(1), target(2), 'o', 'MarkerSize', 15, 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerEdgeColor', [0 0.4470 0.7410]);
elseif strcmp(task, 'butterfly')
    plot(target_Out_pos(1, (start_train+1):end_train), target_Out_pos(2, (start_train+1):end_train), "LineWidth",4);
end
hold on
p = plot([0 Out_test_history_link1(1, i, j) Out_test_history_link2(1, i, j)], [-2 Out_test_history_link1(2, i, j) Out_test_history_link2(2, i, j)], "LineWidth",10);
txt = text(0.5, -2.5, sprintf('%.2f s', (1-start_train)/1000), 'FontSize', 20, 'HorizontalAlignment', 'right');

%Torque gage
t = plot([-1.2 -1.2], [-2.5 -2.5 + 25 * u_history(2, i, j)], "LineWidth",20);
text(-0.85, -2.7, 'Torque', 'FontSize', 15, 'HorizontalAlignment', 'right');

plot([-1.5 -1.5], [-2.5 -2.5 + 25 * 0.15], 'black', "LineWidth",1);
plot([-1.5 -1.55], [-2.5 + 25 * 0.00 -2.5 + 25 * 0.00], 'black', "LineWidth",1);
text(-1.6, -2.5 + 25 * 0.00, '0.00', 'FontSize', 10, 'HorizontalAlignment', 'right');
plot([-1.5 -1.55], [-2.5 + 25 * 0.05 -2.5 + 25 * 0.05], 'black', "LineWidth",1);
text(-1.6, -2.5 + 25 * 0.05, '0.05', 'FontSize', 10, 'HorizontalAlignment', 'right');
plot([-1.5 -1.55], [-2.5 + 25 * 0.10 -2.5 + 25 * 0.10], 'black', "LineWidth",1);
text(-1.6, -2.5 + 25 * 0.10, '0.10', 'FontSize', 10, 'HorizontalAlignment', 'right');
plot([-1.5 -1.55], [-2.5 + 25 * 0.15 -2.5 + 25 * 0.15], 'black', "LineWidth",1);
text(-1.6, -2.5 + 25 * 0.15, '0.15', 'FontSize', 10, 'HorizontalAlignment', 'right');

xlim([-2.0 4.0]);
ylim([-3.5 1.8]);
hold off

vw = VideoWriter('animation_RC_robot.mp4', 'MPEG-4');
vw.Quality = 100;
vw.FrameRate = 50;
open(vw);

frame = getframe(gcf);

for i = 2:n_steps
    p.XData = [0 Out_test_history_link1(1, i, j) Out_test_history_link2(1, i, j)];
    p.YData = [-2 Out_test_history_link1(2, i, j) Out_test_history_link2(2, i, j)];
    txt.String = sprintf('%.2f s', (i-start_train)/1000);

    %Torque gage
    t.YData = [-2.5 -2.5 + 50 * u_history(2, i, j)];

    frame = getframe(gcf);
    writeVideo(vw, frame);
end

close(vw);