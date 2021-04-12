function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward the neural network and return the cost in the

a1=[ones(m,1) X]';
for i=1:1:m
a2(:,i)=sigmoid(Theta1*a1(:,i));
end
a2=[ones(m,1) a2']';
for i=1:1:m
a3(:,i)=sigmoid(Theta2*a2(:,i));
end

Y=zeros(num_labels,m);
for i=1:1:m
    Y(y(i),i)=1;
end
  y=Y;
  
for k=1:1:num_labels
    for i=1:1:m
        JJ(k,i) = -y(k,i).*log(a3(k,i))-((1-y(k,i)).*log(1-a3(k,i)));
    end
end

J=sum(sum(JJ))./m+(lambda./(2.*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.

z2=Theta1*a1;
z3=Theta2*a2;

for i=1:1:m
    for k=1:1:num_labels
        d3(k,i)=a3(k,i)-y(k,i);
    end
end

for i=1:1:m
    for k=1:1:hidden_layer_size
        d22(k,i)=Theta2(:,k+1)'*d3(:,i);
    end
end

for i=1:1:m
    for k=1:1:hidden_layer_size
        d2(k,i)=d22(k,i).*sigmoidGradient(z2(k,i));
    end
end

Theta1_grad=d2*a1'./m;
Theta2_grad=d3*a2'./m;

%
% Part 3: Implement regularization with the cost function and gradients.
%


 for jj=1:1:input_layer_size+1
        if jj==1
              Theta1_grad(:,jj)=Theta1_grad(:,jj);
        else
              Theta1_grad(:,jj)=Theta1_grad(:,jj)+lambda.*Theta1(:,jj)./m;
        end
 end
 
 for jj=1:1:hidden_layer_size+1
        if jj==1
              Theta2_grad(:,jj)=Theta2_grad(:,jj);
        else
              Theta2_grad(:,jj)=Theta2_grad(:,jj)+lambda.*Theta2(:,jj)./m;
        end
 end
 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
