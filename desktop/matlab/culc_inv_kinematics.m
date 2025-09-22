function out = culc_inv_kinematics(xy, L)

l1 = L(1);
l2 = L(2);

x = xy(1,:) + 0;
y = xy(2,:) + 2;
l0 = sqrt(x.^2 + y.^2);

cos_theta2 = -((x.^2 + y.^2) - (l1.^2 + l2.^2)) ./ (2 .* l1 .* l2);
sin_theta2 = sqrt((2 .* l1 .* l2).^2 - (l0.^2 - (l1 .^2 + l2.^2)).^2) ./ (2 .* l1 .*l2);

theta2 = atan2(real(sin_theta2), real((-cos_theta2)));

kc = l1 + l2 .* cos(theta2);
ks = l2 .* sin(theta2);

cos_theta1 = (kc .* x + ks .* y) ./ (kc.^2 + ks.^2);
sin_theta1 = (-ks .* x + kc .* y) ./ (kc .^2 + ks.^2);

theta1 = atan2(sin_theta1, cos_theta1);

out = [theta1; theta2];
end