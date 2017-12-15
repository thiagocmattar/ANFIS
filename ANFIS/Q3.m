%% Sistema Nebulosos: TP3 - QUESTÃO 2
%Sistema de classificação
%Thiago Mattar e Pedro Soares

% Limpa a área de trabalho
clear all; clc;

% Leitura dos dados
load('C:\Users\Thiago\Documents\Sistemas Nebulosos\TP3\ArquivosTP3\dataset_2d.mat')

% Normalização dos dados
x(:,1) = (x(:,1) - mean(x(:,1)))/(max(x(:,1))-min(x(:,1)));
x(:,2) = (x(:,2) - mean(x(:,2)))/(max(x(:,2))-min(x(:,2)));

% Visualização dos dados
figure(1)
plot(x(y==1,1),x(y==1,2),'r+',...
    x(y==0,1),x(y==0,2),'o');
legend('C1','C2');
xlim([-1 1]); ylim([-1 1]);
xlabel('x1'); ylabel('x2');
title('Distribuição dos dados normalizados');

% Definição de Y
y(y==0) = -1;

%% Definindo treino e teste 
[trainX,trainY,testX,testY] = SplitTrainAndTest(x,y);

%% Definição do n° de regras

% Função de parametrização
fprintf('\n Parametrizando modelo ... \n');
accuracy = testModel(trainX,trainY,30);

figure(2)
subplot(2,1,1);
plot(accuracy(2:end,1),'LineWidth',2);
title('Avaliação do número de regras');
xlim([2 29]); xlabel('N° de regras'); ylabel('Acurácia média'); grid;
subplot(2,1,2);
plot(accuracy(2:end,2),'r','LineWidth',2);
xlim([2 29]); xlabel('N° de regras'); ylabel('Variância'); grid;



%% Definindo dimensão final do ploblema
[~,nclus_best] = max(accuracy(:,1)./accuracy(:,2).^2);
fprintf('\n Calculando solução final ...');

% Solução final
acc_fin = zeros(30,1);
for i = 1:30
    [trainX,trainY,testX,testY] = SplitTrainAndTest(x,y);
    [acc_fin(i),~] = trainAndTestAnfisByFCM(trainX,trainY,nclus_best,testX,testY);  
end

acc_mean = mean(acc_fin);
acc_sd = std(acc_fin);

fprintf('\n\n');
disp('-----------SOLUÇÃO FINAL--------'); 
disp(['N° de regras: ' num2str(nclus_best)]);
disp(['Acurária: ' num2str(acc_mean) ' +/- ' num2str(acc_sd)]);

