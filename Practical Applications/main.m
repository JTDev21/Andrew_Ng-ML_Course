% Read and Store Training examples
data = dlmread('TSLA.csv');

trainingSetSize = 8000;
trainingSet = data(1:trainingSetSize, :);

% Assignment
x = [trainingSet(:, 1:3)];
y = [trainingSet(:, 4)];
m = size(x,1);

X = [ones(m,1), x];

theta = zeros(size(X, 2), 1);

% Normal Equation
theta = pinv(X' * X) * X' * y;

% Cost function for the model
prediction = X * theta;
squareErrors = (prediction - y) .^ 2;
J = (1 / (2*m)) * sum(squareErrors);
fprintf('[Training model] Cost computed:  %d\n', J);



% %======================Predict Cost===================
testData = data((trainingSetSize + 1): size(data, 1), :);
X_test = [ones(size(testData, 1), 1), testData(:, 1:3)];
y_test = testData(:, 4);

predictedCost = X_test * theta;
for i = 1:size(predictedCost,1)
    fprintf('Predicted Cost: $%d | Actual Cost: $%d | Error: %d | Square error: %d\n', predictedCost(i), y_test(i), predictedCost(i) - y_test(i), (predictedCost(i) - y_test(i))^2);
endfor;

% hold on;
% idx = [0:1:size(data,1)-1]';
% plot(idx, data(:, 4));
% plot(idx(trainingSetSize+1:size(data,1)), data(trainingSetSize+1: size(data,1),:)*theta, '*')