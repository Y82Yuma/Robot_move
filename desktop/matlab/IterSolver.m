function X = IterSolver(Out, WOut, X0, iter)

[numOut, numModules] = size(WOut);
X = X0;

for i = 1:iter
    for j = 1:numOut
        for k = 1:numModules
            X(k) = (Out(j) - (WOut(j, :) * X' - WOut(j, k) * X(k))) / WOut(j, k);
            if X(k) > 1
                X(k) = 0.999;
            elseif X(k) < -1
                X(k) = -0.999;
            end
        end
    end
end

end