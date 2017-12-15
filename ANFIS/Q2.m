%% Sistema Nebulosos: TP3 - QUESTÃO 2
%Série temporal caótica - ANFIS
%Thiago Mattar e Pedro Soares

% Limpa a área de trabalho
clear all; clc;

% Leitura dos dados
cd('C:\Users\Thiago\Documents\Sistemas Nebulosos\TP3\ArquivosTP3')
load mgdata.csv
cd('C:\Users\Thiago\Documents\Sistemas Nebulosos\TP3\')

%% Definição dos dados
n = length(mgdata);
range = 19:(n-6);
x = zeros(length(range),4);
y = zeros(length(range),1);
for i=range
    x(i-18,:) = [mgdata(i-18) mgdata(i-12) mgdata(i-6) mgdata(i)];
    y(i-18) = mgdata(i+6);
end

%% Separação em treino e teste
r = 0.8;
splitIdx = int64(1:(r*n));
trainX = x(splitIdx,:);
trainY = y(splitIdx);
testX = x(splitIdx(end):end,:);
testY = y(splitIdx(end):end);

%% Definição dos dados e modelo
anfisInputs = [trainX trainY];

% Definição das configurações
opt = anfisOptions();
opt.InitialFIS = 2;
opt.EpochNumber = 100;

% Treinamento
[fis,trainError] = anfis(anfisInputs,opt);

% Teste
Yhat = evalfis(testX,fis);

% Plot da aproximação
figure(1)
plot(testY,'LineWidth',2); hold on
plot(Yhat,'r--','LineWidth',2)
title('Aproximação de f(x) - ANFIS');
ylim([0.4 1.5])
legend('Training Data','ANFIS Output','Location','NorthEast')
xlabel('k'); ylabel('y'); grid;

% Plot do erro
figure(2)
plot(trainError,'k','LineWidth',2); hold on
title('Erro de treinamento');
xlabel('epoch'); ylabel('RMSE'); grid;

