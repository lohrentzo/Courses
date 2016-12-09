function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%disp(size(X))
%disp(X)

best = Inf;

for i = 1:size(X,1)
  x = X(i, :);
  best = Inf;
  for u = 1:K
    mu = centroids(u, :);
    
    %fprintf('x: ');
    %disp(x)
    
    %fprintf('mu: ');
    %disp(mu)
    
    %fprintf('x-mu: ');
    %disp(x-mu)
    
    x_minus_mu = x - mu;
    
    dist = sum(x_minus_mu .^ 2);
    %disp(dist)
    
    %d = dot(x_minus_mu, x_minus_mu);
    %disp(d)
    
    if dist < best
     %fprintf('i: %d, u: %d, dist: ', i, u);
     %disp(dist)
     
      best = dist;
      idx(i) = u;
    end
  end
end




% =============================================================

end

