%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%       Intelligent Systems         %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%        NeuralNetwork 1.2          %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%           Utsav Shah              %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;
% tic;
%% Loading Data

data.image.all = dlmread('MNISTnumImages5000.txt');
data.label.all = dlmread('MNISTnumLabels5000.txt');

%% Defining Train and Test Data

data.image.train = data.image.all(1:4000,:);
data.image.test = data.image.all(4001:end,:);

data.label.train = data.label.all(1:4000,:);
data.label.test = data.label.all(4001:end,:);
%%
% Initializing Matrices

w_jk = 0.2 .* rand(200,785) - 0.1;      % hidden layer - weight from input k to hidden neuron j
w_ij = 0.2 .* rand(784,201) - 0.1;      % output layer - weight from hidden neuron j to output i      
eta = 0.02;      % learning rate
alpha = 0.2;    % momentum
% dw_ij=zeros(784,201,4000);
% dw_jk=zeros(200,785,4000);
epoch=1;

while epoch==1 || J2(epoch-1)>=25000 % low value is 50 from previous
    shuffle_indices=randperm(4000);
    image_data=data.image.train(shuffle_indices,:);
    J2(epoch)=0;
    for q = 1:4000
        x = image_data(q,:).';      % n = all columns = 784
        y = x;       % actual output

        % Calculating s for hidden layer
        
        s_j = w_jk * [1;x];       % hidden layer
        h = perceptron(s_j);    % hidden layer output which will be given to output layer input
        s_i = w_ij * [1;h];       % output layer
        y_hat = perceptron(s_i);       % calculated output
        
        error = y - y_hat;        % calcuting error
        J2(epoch) = J2(epoch) + (sum(error).^2);        % loss function    
        
        % Back Propogation Algorithm 
        delta_i = error .* diffy(s_i);
        if q==1
            dw_ij = eta * delta_i*[1;h].';
        else
            dw_ij = eta * delta_i*[1;h].' + alpha * dw_ij;
        end

        delta_j=diffy(s_j) .* (w_ij(:,2:end).'*delta_i);
        if q==1
            dw_jk=eta*delta_j*[1;x].';
        else
            dw_jk = eta*delta_j*[1;x].' + alpha*dw_jk;
        end
        w_ij=w_ij+dw_ij;
        w_jk=w_jk+dw_jk;
    end
    epoch=epoch+1;
end
% J2_train=sum(J2);
%% Testing, plotting outside loop
y_hat_test=zeros(784,1000);
J2_test=zeros(1000,1);
for q = 1:1000
    x_test = data.image.test(q,:).';      % n = all columns = 784
    y_test = x_test;
    % Calculating s for hidden layer
    s_j_test = w_jk * [1;x_test];       % hidden layer
    h_test = perceptron(s_j_test);      % hidden layer output which will be given to output layer input
    s_i_test = w_ij * [1;h_test];       % output layer
    y_hat_test (:,q)= perceptron(s_i_test);       % calculated output
    error_test = y_test-y_hat_test(:,q);
    J2_test (q) = sum((error_test).^2);
end
J2_test = sum(J2_test);

% t = toc;
%%
figure(1)
bar([J2(end),J2_test],0.2);
ylabel('J_2 error');
set(gca,'XTickLabel',{'Training Set','Test Set'})
title('\bf{Loss Function Levels}')
%%
figure(2)
w1=w_jk(:,2:end);
index=reshape(1:200,20,10).';
for i=1:10
    for j=1:20
        subplot(10,20,index(i,j))
        imshow(mat2gray(reshape(w1(index(i,j),:),28,28).'))
        axis off
        title(sprintf('%i',index(i,j)));
    end
end
suptitle('\bf{Weights between input layer and hidden layer neurons (number in title)}')
%% Auto Encoder Network
plotx=zeros(28,28,1000);
ploty=zeros(28,28,1000);
for qqq=1:1000
        plotx(:,:,qqq)=reshape(data.image.test(qqq,:),28, 28);
        temp=y_hat_test((row-1)*28+1:row*28,qqq).';
        ploty(:,:,qqq)=reshape(temp,28,28);
end
%
% Plotting 1st 100 test images,input
index=reshape(1:100,20,10).';
figure(3)
for i=1:10
    for j=1:10
        subplot(10,10,index(i,j))
        imshow(mat2gray(plotx(:,:,index(i,j))))
        axis off
    end
end

% Plotting 1st 100 test images,output
figure(4)
for i=1:10
    for j=1:10
        subplot(10,10,index(i,j))
        imshow(mat2gray(ploty(:,:,index(i,j))))
        axis off
    end
end


[yyy,Fs] = audioread('Tring.mp3');
sound(yyy,Fs);