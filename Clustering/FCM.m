clear all;
close all;
clc;

% carregando e plotando os dados
load fcm_dataset.mat;

k = 4; %nº de clusters
n = size(x,1);
U = zeros(n,k);

% figure(1);
% hold on;
% plot(x(:,1), x(:,2),'b.');
% grid on;

%Inicializando matriz de pertinência
for i = 1:n
    aux = rand(1,4);
    aux = aux/sum(aux);
    U(i,:) = aux;
end

Uinit = U;
cent30 = zeros(30,8);
Jfinal = zeros(30,1);

for z = 1:30
    
    U = Uinit;
    centroids = zeros(k,2);
    iter = 0;
    changes = true;
    
    while(changes)
        
        %1. Cálculo dos centróides
        centroids = zeros(k,2);
        for i = 1:k
            centroids(i,1) = sum((U(:,i).^2).*x(:,1))/sum(U(:,i).^2);
            centroids(i,2) = sum((U(:,i).^2).*x(:,2))/sum(U(:,i).^2);
        end
        
        % ploting the new centroids
        %             xdata = centroids(:,1);
        %             ydata = centroids(:,2);
        %             pause(1)
        %             h = plot(xdata, ydata, 'ko', 'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','g', 'MarkerSize', 10);
        %             set(h,'YDataSource','ydata')
        %             set(h,'XDataSource','xdata')
        %             refreshdata
        
        %2. Computando a função de custo
        Jaux = 0;
        for i = 1:n
            for j = 1:k
                Jaux = Jaux + U(i,j)*norm(x(i,:)-centroids(j,:))^2;
            end
        end
        iter = iter+1;
        J(iter) = Jaux;
        
        %3. Computando a nova matriz U
        for i = 1:n
            for j = 1:k
                if(x(i,1)==centroids(j,1) && x(i,2)==centroids(j,2))
                    U(i,:) = 0;
                    U(i,j) = 1;
                else
                    dist = norm(x(i,:)-centroids(j,:));       %(x(i,:)-centroids(j,:))*(x(i,:)-centroids(j,:))';
                    total.dist = 0;
                    for aux = 1:k
                        total.dist = total.dist + ...
                            norm(x(i,:)-centroids(aux,:));       %(x(i,:)-centroids(aux,:))*(x(i,:)-centroids(aux,:))';
                    end
                    U(i,j) = 1/(dist/total.dist)^2;
                end
            end
        end
        
        for i = 1:n
            U(i,:) = U(i,:)/sum(U(i,:));
        end
        
        %             indexes = zeros(n,1);
        %             for i = 1:n
        %                 [~,idx] = max(U(i,:));
        %                 indexes(i) = idx;
        %             end
        
        %             figure(1)
        %             plot(x(indexes==1,1),x(indexes==1,2),'b.');
        %             hold on;
        %             plot(x(indexes==2,1),x(indexes==2,2),'r.');
        %             hold on;
        %             plot(x(indexes==3,1),x(indexes==3,2),'c.');
        %             hold on;
        %             plot(x(indexes==4,1),x(indexes==4,2),'m.');
        %             grid on;
        
        if(iter>1)
            if ((abs(J(iter)-J(iter-1)))<0.001)
                changes = false;
            end
        end
        
    end
    
    
    % ploting the objective function as a function of the number of iterations
    %         figure(2);
    %         hold on;
    %         plot(1:(iter-1), J(2:iter), 'b--', 'LineWidth', 2);
    %         grid on;
    
    Jfinal(z) = J(iter);
    cent30(z,:) = centroids(:);
    
    [Jfinal(z), z,iter]
    fprintf('\n')
    
    
end
