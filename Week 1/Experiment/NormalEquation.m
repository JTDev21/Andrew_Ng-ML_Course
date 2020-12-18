x = [1 2 3 4 5]; 
y = [1 2 3 4 5];
x = x';
y = y';
##data = dlmread ('Class_Data.csv');
##data(1,:) = [];
##x = data(:,2);
##y = data(:,1);

X = [ones(length(x), 1), x];

theta = pinv(X' * X) * X' * y

##cost = (1/(2 * length(y))) * sum(((theta(1) + theta(2) * x) - y) .^ 2)

##hold on;
##loglog(x,y, '*')
###scatter(x,y);
##plot(x, theta(1) + theta(2) * x)
##hold off;