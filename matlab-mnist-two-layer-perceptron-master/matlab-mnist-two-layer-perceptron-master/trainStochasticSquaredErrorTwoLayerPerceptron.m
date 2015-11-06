function [hiddenWeights, outputWeights, error, startTime, finalTime] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate,name)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
%   
    %start time
    startTime = clock;
    
    % The number of training vectors.//60000
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);
    inputVector(1:784,1:batchSize) = 0;
    figure; hold on;
    
    for k = 1: batchSize
        n(k) = k;%floor(rand(1)*trainingSetSize + 1);
            
        % Propagate the input vector through the network.
        inputVector(:,k) = inputValues(:, n(k));
        
        blr = inputVector(:,k);
        blur = reshape(blr,28,28,1);
            
        H = fspecial('disk',3);
        blurred = imfilter(blur,H,'replicate');
        inputVector(:,k) = reshape(blurred,784,1);
    end
    
    for t = 1: epochs
        fprintf('Treinamento em %d de 100 \n', (t/epochs)*100);
        
        %learningRate = 0.1 - (t*0.0005);
        
        for k = 1: batchSize
            % Select which input vector to train on.

            hiddenActualInput = hiddenWeights*inputVector(:,k);
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);

            outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector';
            hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector(:,k)';
        end;
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector(:,k) = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector(:,k))) - targetVector, 2);
        end;
        error = error/batchSize;
        
        plot(t, error,'*');
        
    end;
    %salva imagem do grafico
    %print(['Graficos/Treino',name,'.png'],'-dpng');
    finalTime = clock;
end