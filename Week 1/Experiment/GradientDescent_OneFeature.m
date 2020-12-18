x = [1 2 3 4 5]; 
y = [1 2 3 4 5];
m = length(y);

iterations = 2000;
alpha = 0.001;

theta0 = 0;
theta1 = 0;

for i = 1:iterations
    temp0 = theta0 - (alpha * (1/m)) * (sum(((theta0 + theta1 .* x) - y)));
    temp1 = theta1 - (alpha * (1/m)) * (sum(((theta0 + theta1 .* x) - y) .* x));
    
    theta0 = temp0;
    theta1 = temp1;
    
end
theta0
theta1

