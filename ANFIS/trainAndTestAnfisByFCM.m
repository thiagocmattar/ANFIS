function [acc,ANFIS] = trainAndTestAnfisByFCM(trainX,trainY,k,testX,testY)

      
    %Inserindo amostras artificiais para evitar o problema de range
    trainX = [trainX; 1 1; -1 -1];
    trainY = [trainY; 1; 0];

    % Definindo FIS inicial com base no FCM
    opt = genfisOptions('FCMClustering','FISType','sugeno');
    opt.NumClusters = k;
    opt.Verbose = 0;
    opt.MinImprovement = 0.000001;
    opt.MaxNumIteration = 200;
    InitialFis = genfis(trainX,trainY,opt);
    
    % Definindo estrutura ANFIS
    ANFISOpt = anfisOptions('InitialFIS',InitialFis);
    ANFISOpt.EpochNumber = 100;
    ANFISOpt.DisplayANFISInformation = 0;
    ANFISOpt.DisplayErrorValues = 0;
    ANFISOpt.DisplayStepSize = 0;
    ANFISOpt.DisplayFinalResults = 0;
    ANFISOpt.OptimizationMethod = 1;

    % Treinamento
    anfisInputs = [trainX trainY];
    ANFIS = anfis(anfisInputs,ANFISOpt);
    
    % Teste
    anfisOutput = evalfis(testX,ANFIS);
    anfisOutput(anfisOutput>0) = 1;
    anfisOutput(anfisOutput<=0) = -1;
    
    % Acurácia
    acc = sum(diag(confusionmat(anfisOutput,testY)))/length(testY);
    
end