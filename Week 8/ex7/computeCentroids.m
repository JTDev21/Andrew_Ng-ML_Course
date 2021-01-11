function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% %======================== Version 1 ========================
% for k = 1:K
%     k_idx = find(idx == k); % Get the index of values that matches k
%     Ck = size(k_idx, 1); % Get the size of k_idx
%     mu_k = (1/Ck) * sum(X(k_idx, :)); % Computing centoid means 
%     centroids(k, :) = mu_k;
% endfor;

% %======================== Version 2 ========================
% % Error: nonconformant arguments (op1 is 1x3, op2 is 0x1) on 2nd k-means test
%     % Fixed: Complete kMeansInitCentroids.m
% for k = 1:K
%     x_idx = find(idx == k);
%     centroids(k,:) = mean(X(x_idx, :));
% endfor;

% %======================== Version 2 - Vectorized Implementation ========================
% https://www.coursera.org/learn/machine-learning/discussions/weeks/8/threads/CHrCBRe9EeevPg7X4ZMqyg
% Note: I believe the vectorized implementation was implemented incorrectly
selector = eye(K)(idx,:); % Logical boolean vector
add_xi_K = selector' * X; % Add the examples that are assigned to centroid k
num_Ck = sum(selector)'; % Get |Ck| the number of examples in the set assigned to centroid k

% 2 ways to divide
% centroids = add_xi_K ./ num_Ck;
centroids = bsxfun(@rdivide, add_xi_K, num_Ck);

% % One line code
% centroids = (eye(K)(idx,:)' * X) ./ sum(eye(K)(idx,:))';

% =============================================================


end

