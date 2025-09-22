%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. construct recurrent neural networks %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('constructing RNN:');

% input connections WIn
WIn = input_weight_amp * randn(numUnits, numIn);

% feedback connections WFb
WFb = feedback_weight_amp * randn(numUnits, numOut);

% initial readout connections WOut(postsyn,presyn)
WOut = zeros(numOut, numUnits);

% P matrix for readout learning
P = eye(numUnits) / delta;

%% recurrent connections W
W_mask = rand(numUnits, numUnits);
W_mask(W_mask <= p_connect) = 1;
W_mask(W_mask < 1) = 0;
W = randn(numUnits, numUnits) * scale;
W = W .* W_mask;
Wk = W;
Wk(logical(eye(numUnits))) = 0;	% set self-connections to zero
W = sparse(Wk);

% performance (Peason's correlation coefficient)
R2_learn = zeros(numOut, n_learn_loops);
R2_test = zeros(numOut, n_test_loops);

% Store history
Out_learn_history = zeros(numOut, n_steps, n_learn_loops);
Out_test_history = zeros(numOut, n_steps, n_test_loops);
Out_test_history_link2 = zeros(numOut, n_steps, n_test_loops);
Out_test_history_link1 = zeros(numOut, n_steps, n_test_loops);
error_history = zeros(numOut, n_steps, n_test_loops);
u_history = zeros(numOut, n_steps, n_test_loops);