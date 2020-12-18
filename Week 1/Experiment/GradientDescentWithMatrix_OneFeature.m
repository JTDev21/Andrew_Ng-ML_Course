x = [1 2 3 4 5]; 
y = [1 2 3 4 5];
m = length(y);

X = [ones(m, 1), x'];
theta = zeros(2,1);

iterations = 2000;
alpha = 0.001;

for i = 1:iterations
  theta = theta - (alpha * (1/m) * ((X * theta)' - y) * X)';
end
theta