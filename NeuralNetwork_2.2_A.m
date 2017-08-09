%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%       Intelligent Systems         %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%      NeuralNetwork 2.2 - A        %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%           Utsav Shah              %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;
%% Loading Data

data.image.all = dlmread('MNISTnumImages5000.txt');
data.label.all = dlmread('MNISTnumLabels5000.txt');
%% Defining Train and Test Data

data.image.train = data.image.all(1:4000,:);
data.image.test = data.image.all(4001:end,:);

data.label.train = data.label.all(1:4000,:);
data.label.test = data.label.all(4001:end,:);
%% Taking Number of Layers as Input from User
j=1;
while j<=0 && ~isinteger(j)
    j = input('How many hidden layers you need \n');
end
%% Initializing Matrices
tic;
% w_jk = 0.2 .* rand(200,785) - 0.1;  % hidden layer - weight from input k to hidden neuron j
load hw3_problem2.mat
w_ij = 0.2 .* rand(10,201) - 0.1;   % output layer - weight from hidden neuron j to output i
eta = 0.2;                          % learning rate
alpha = 0.4;                        % momentum
error=zeros(10,3500);               % error_test = zeros(10,1000);
J=zeros(1,4000);
hit_rate_train=zeros(50,1);
confusion_train=zeros(10,10);
flag=0;                             % J2 error flag, 0 => error is not below threshold
for epoch = 1:500
    shuffle_indices=randperm(4000);
    image_data=data.image.train(shuffle_indices,:);
    label_data=data.label.train(shuffle_indices,:);
    y=zeros(10,3500);
    hits_train=0;
    % Presenting a subset of training set
    for q = 1:3500
        % Feed forward
        x = image_data(q,:).';              % n = all columns = 784+1(later)
        y(label_data(q,:) + 1,q) = 1;       % actual output
        % Calculating s for hidden layer
        s_j = w_jk * [1;x];
        h = perceptron(s_j);            % hidden layer output which will be given to output layer input
        s_i = w_ij * [1;h];             % output layer
        y_hat = perceptron(s_i);        % calculated output l=10
        error(:,q) = y(:,q) - y_hat;                % error(10x1)
        J(q) = 0.5 * (sum(error(:,q)).^2);          % loss function
%         if J(q)<=1e-20 && J(q-1)<=1e-20
%             flag=1;
%             break
%         end
        % Calculating hit_rate performance
        if length(find(y_hat==max(y_hat)))==1 ...
                && find(y_hat==max(y_hat))-1==label_data(q,:)
            hits_train=hits_train+1;
        end
        % Back Propogation 
        % output layer delta
        delta_i = error(:,q) .* diffy(s_i);         
        % Applying Momentum
        if q==1
            dw_ij = eta * delta_i*[1;h].';
        else
            dw_ij = dw_ij + alpha * dw_ij;
        end
        
%         % hidden layer delta
%         delta_j=diffy(s_j) .* (w_ij(:,2:end).'*delta_i);
%         if q==1
%             dw_jk=eta*delta_j*[1;x].';
%         else
%             dw_jk = dw_jk + alpha*dw_jk;
%         end
        
        % Weight updation
        w_ij=w_ij+dw_ij;
%         w_jk=w_jk+dw_jk;
        % Confusion Matrix for Train Data
        confusion_train(label_data(q,:) + 1,find(y_hat==max(y_hat)))=...
            confusion_train(label_data(q,:) + 1,find(y_hat==max(y_hat)))+1;
    end
    
    % hit rate for training data
    hit_rate_train(epoch)=hits_train*100/q;
%     if flag==1
%         break
%     end
end
%% Testing using the last 1000 datapoints, Feed forward only
error_test=zeros(10,1000);
y_test=zeros(10,1000);
hits_test=0;
confusion_test=zeros(10,10);
for q = 1:1000
    x_test = data.image.test(q,:).';        % n = all columns = 784
    y_test(data.label.test(q,:)+1,q) = 1;   % generating y for label data
    s_j_test = w_jk * [1;x_test];           % hidden layer s
    h_test = perceptron(s_j_test);          % hidden layer output which will be given to output layer input
    s_i_test = w_ij * [1;h_test];           % output layer s
    y_hat_test = perceptron(s_i_test);      % calculated output
    error_test(:,q)= y_test(:,q) - y_hat_test;    % error    
    % Calculating hit_rate performance
    if length(find(y_hat_test==max(y_hat_test)))==1 ...
            && find(y_hat_test==max(y_hat_test))-1==data.label.test(q,:)
        hits_test=hits_test+1;
    end    
    % Confusion Matrix for Test Data
    confusion_test(data.label.test(q,:) + 1,find(y_hat_test==max(y_hat_test)))=...
        confusion_test(data.label.test(q,:) + 1,find(y_hat_test==max(y_hat_test)))+1;
        
end
%% Store data for testing epochs for plotting
% hit rate for testing data
hit_rate_test=hits_test*100/1000;
t = toc;
%% Tring Tring
[yyy,Fs] = audioread('Tring.mp3');
sound(yyy,Fs);
%% Plots
% Training set hit rate, test set hit rate as a line
figure(1)
plot(1:length(hit_rate_train),1-(hit_rate_train)/100,...
    1:length(hit_rate_train),ones(500)*1-hit_rate_test/100,'r')
xlabel('Epoch');
ylabel('(1 - Hit Rate)');
title('\bf{Hit Rate Performance}');
legend('Training Hit Rate Performance vs Epoch','Test Hit Rate Performance (Single Value)')
% Training Confusion Matrix
figure(2)
b1=bar3(confusion_train);
xlabel('Target')
ylabel('Actual')
zlabel('Number classified as')
digits={'0','1','2','3','4','5','6','7','8','9'};
set(gca,'XTickLabel',digits,'YTickLabel',digits)
title('\bf{Training Confusion Matrix}')
colorbar
for k = 1:length(b1)
    zdata = b1(k).ZData;
    b1(k).CData = zdata;
    b1(k).FaceColor = 'interp';
end
view([90 0])
% Test Confusion Matrix
figure(3)
b2=bar3(confusion_test);
xlabel('Target')
ylabel('Actual')
zlabel('Number classified as')
digits={'0','1','2','3','4','5','6','7','8','9'};
set(gca,'XTickLabel',digits,'YTickLabel',digits)
title('\bf{Test Confusion Matrix}')
for k = 1:length(b2)
    zdata = b2(k).ZData;
    b2(k).CData = zdata;
    b2(k).FaceColor = 'interp';
end
view([90 0])
colorbar 
colormap jet