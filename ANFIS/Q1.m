%% Sistema Nebulosos: TP3 - QUESTÃO 1
%Aproximação da função seno - ANFIS
%Thiago Mattar e Pedro Soares

% Limpa a área de trabalho
clear all; clc;

% Definição dos dados
x = (0:0.1:(2*pi))';
y = sin(x);
anfisInputs = [x y];

% Definição das configurações
opt = anfisOptions();
opt.InitialFIS = 3;
opt.EpochNumber = 50;

% Treinamento
fis = anfis(anfisInputs,opt);

%Avaliação dos dados
anfisOutput = evalfis(x,fis);

%Plot da aproximação
figure(1)
plot(x,y,'LineWidth',2); hold on
plot(x,anfisOutput,'r--','LineWidth',2)
title('Aproximação de f(x) por Sugeno');
xlim([0 2*pi]); ylim([-1.1 1.1])
legend('Training Data','ANFIS Output','Location','NorthEast')
xlabel('x'); ylabel('y'); grid;

%Plot das funções de pertinencia
figure(2)
plotmf(fis,'input',1)
title('Funções de pertinência'); grid;

%Print do RMSE
e2 = sum((y - anfisOutput).^2);
disp(['RMSE = ' num2str(e2)]);






