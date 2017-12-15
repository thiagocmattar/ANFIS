clear all;
close all;
clc;

% carregando e plotando os dados
load SyntheticDataset.mat;
X = x;

% figure(1);
% hold on;
% plot(X(:,1), X(:,2),'b.');
% grid on;

%------------------------------------------------------------------------
% Algoritmo K-Means

% define o numero de grupos
K = 4;

% step 1: randomly assign a cluster to each one of the patterns

Jit = zeros(50,1);
it = zeros(50,1);
for z = 1:50
    n = size(X,1);
    U = zeros(n,K);    % partition matrix
    idx = zeros(n,1);
    for i = 1:n,
        rnd = randi(K);
        U(i,rnd) = 1;
        idx(i) = rnd;
    end
    
    % calculating the objective function
    W = zeros(K,1);
    for j = 1:K,
        indexes = find(idx==j);
        Clusj = X(indexes,:);
        W(j) = (1/length(indexes)) * sum(pdist(Clusj)); % pdist calculates the n(n-1)/2 distances among all patterns in the Cluster K
    end
    J(1) = sum(W);
    
    changes = true;
    oldIdx = idx;
    iter = 1;
    while (changes)    % iterate until the cluster assignments stop changing
        
        
        % computing the initial centroids
        centroids = zeros(K,2);
        
        for j = 1:K,
            Gj = U(:,j);
            onesIndexes = find(Gj == 1);
            Xj = X(onesIndexes,:);
            centroids(j,:) = mean(Xj);
        end;
        
        %ploting the new centroids
%             xdata = centroids(:,1);
%             ydata = centroids(:,2);
%             pause(1)
%             h = plot(xdata, ydata, 'ko', 'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','g', 'MarkerSize', 10);
%             set(h,'YDataSource','ydata')
%             set(h,'XDataSource','xdata')
%             refreshdata
        
        
        % assign each pattern to the cluster whose centroid is closest
        U = zeros(n,K);
        for i = 1:n,
            pattern = X(i,:);
            smallDistance = inf;
            for j = 1:K,
                gc = centroids(j,:);
                distance = sum((pattern-gc).^2);  % squared Euclidian distance from pattern to each centroid
                if (distance < smallDistance),
                    smallDistance = distance;
                    smallIndex = j;
                end
            end
            U(i,smallIndex) = 1;
            idx(i) = smallIndex;
        end
        
        % calculating the objective function
        indexes = zeros(n,1);
        clus = unique(idx);
        c = length(clus);
        W = zeros(c,1);
        for j = 1:c,
            indexes = find(idx==clus(j));
            Clusj = X(indexes,:);
            W(j) = 1/length(indexes) * sum(pdist(Clusj)); % pdist calculates the n(n-1)/2 distances among all patterns in the Cluster K
        end
        iter = iter + 1;
        J(iter) = sum(W);
        
        % verifying the stop criteria
        if isequal(idx,oldIdx),
            changes = false;
        else
            oldIdx = idx;
        end;
        
    end;
    
    % ploting the final clustering resulting from K-Means
    clus = unique(idx);
    colors = {'b.', 'r.', 'c.', 'm.', 'y.', 'k.'};
    for i = 1:length(clus),
        
        indexes = find(idx==clus(i));
        plot(X(indexes,1), X(indexes,2), colors{i});
        
    end
    
    Jit(z) = J(iter);
    it(z) = iter;
    out = ['Iteração:',num2str(it(z)),';     J=',num2str(Jit(z))];
    disp(out);
end
% ploting the objective function as a function of the number of iterations
% figure(2);
% hold on;
% plot(1:iter, J(1:iter), 'b--', 'LineWidth', 2);
% grid on;

N = sum(Jit>mean(Jit));
m = mean(it);

