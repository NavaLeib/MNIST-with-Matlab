%% Analyzing Complex Systems   - Exercise 1 - Neural Network

%% Nava Leibovich . ID# 038212056. 

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%  (displayData.m)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('Targil1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


% Plotting the correpsonding y's --> To verify the adequacy  
yy=reshape(mod(y(sel),10),10,10)';
yplot=num2str(yy);
annotation(figure(2),'textbox',[0.159260869565217 0.106094808126411 0.642478260869565 0.817155756207675],'string',yplot,'FontSize',24, 'FitBoxToText','off');

fprintf('Program paused. Press enter to continue.\n');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following part was written for validation and may remain a comment:

 %% ================ Validation of the Potential Function =============
 %% =====With/Without Regularization (Comparing to known values)=======
 %% =============and the Sigmoid function and its Gradient=============

% 
% fprintf('\nLoading Saved Neural Network Parameters ...\n')
% 
% % Load the weights into variables Theta1 and Theta2
% load('weights.mat');
% 
% % Unroll parameters 
% nn_params = [Theta1(:) ; Theta2(:)];

%  Implementing the feedforward to compute the cost to verify that
%  the implementation is correct by verifying that I get the same cost
%  as us for the fixed debugging parameters.

%
% fprintf('\nFeedforward Using Neural Network ...\n')
% 
% % Weight regularization parameter (we set this to 0 here).
% lambda = 0;
% 
% J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                    num_labels, X, y, lambda);
% 
% fprintf(['Cost at parameters (loaded from ex4weights): %f '...
%          '\n(this value should be about 0.287629)\n'], J);
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

% Chacking the potentil function after adding Regularization
%

% fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
% 
% % Weight regularization parameter (we set this to 1 here).
% lambda = 1;
% 
% J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                    num_labels, X, y, lambda);
% 
% fprintf(['Cost at parameters (loaded from ex4weights): %f '...
%          '\n(this value should be about 0.383770)\n'], J);
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ================ Part 2: Chaking the Gradients ================
%  I implment a two  layer neural network that classifies digits. 
%  First implementing a function to initialize the weights of the 
%  neural network  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% ++++ Part 2a: Comparing between Numeric and BackPropagation ++++


fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


fprintf('\nChecking Backpropagation (with Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% =================== Part 3: Training NN ===================
%  We will now use "fmincg". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

% The predict value of X
pred = predict(Theta1, Theta2, X);

%This lets me compute the training set accuracy (in percent).
fprintf('\nTraining Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ========== Part 3a: Training NN with 60% from the data=============
%% ===================and Cross Validation data=======================

fprintf('\nTraining Neural Network with 60 percent from the as training set.. \n')

 
trn=randperm(size(X,1));
trn1=trn(1:3000);
TrainingSetX = X(trn1,:);
TrainingSetY = y(trn1);
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 200);

j=1;
for lambda = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10];

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, TrainingSetX,TrainingSetY, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Cost=cost;




Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% Cross Validation Data
trn2=trn(3001:4000);
CrossValX=X(trn2,:);
crossValY=y(trn2);

% The predict value of X
pred = predict(Theta1, Theta2, CrossValX);
predTraning = predict(Theta1, Theta2, TrainingSetX);

%This lets me compute the training set accuracy (in percent).
%fprintf('\nTraining Accuracy: %f\n', mean(double(pred == CrossValY)) * 100);

ErrorCV(j)= mean(1-double(pred == CrossValY));
Error(j)= mean(1-double(predTraining == TrainingSetT));

j=j+1;
end

lambda = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10];
plot(lambda,ErrorCV,lambda,Error);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ====================Part 3c: Test (Unseen) Data ==================
% in this part I use the optimal regularization and MaxIter parameters 
% for the digits recogntion on unseen data 

lambda= 1
options = optimset('MaxIter', 100);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, TrainingSetX,TrainingSetY, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Cost=cost;




Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% Cross Validation Data
trn3=trn(4001:5000);
UnseenX=X(trn3,:);
UnseenY=y(trn3);

% The predict value of X
predTest = predict(Theta1, Theta2, UnseenX);


%This lets me compute the training set accuracy (in percent).
fprintf('\nTraining Accuracy: %f\n', mean(double(predTest == UnseenY)) * 100);



