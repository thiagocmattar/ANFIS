function acc = testModel(X,Y,kmax)

    %Dimensão do problema
    n = kmax;
    acc = zeros(n,2);

    for i = 2:kmax
        
        %Vetor de acurácias
        acc_mean = zeros(30,1);
        k = i;
        
        %Tempo
        t0 = clock;
        
        for j = 1:50
            %Definindo dados de treino e teste
            [trainX,trainY,testX,testY] = SplitTrainAndTest(X,Y);

            %Treinando e testando modelo
            [acc_mean(j),~] = trainAndTestAnfisByFCM(trainX,trainY,k,testX,testY);           
        end
        
        acc(i,1) = mean(acc_mean);
        acc(i,2) = std(acc_mean);
        
        t1 = clock;
        disp(['Iteration ' num2str(i) ' of ' num2str(kmax) ...
         ', elapsed time: ' num2str(etime(t1,t0)) ' seconds, Acc = ' num2str(acc(i,1))]);
    
    end
end