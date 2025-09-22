function xy = culc_kinematics_elbow(theta, L)

theta1 = theta(1, :);
theta2 = theta(2, :);

l1 = L(1);
l2 = L(2);

xy = [0 + l1 * cos(theta1);...
    -2 + l1 * sin(theta1)];
end