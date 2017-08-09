# neural-networks

Project:
Real-world application sized Neural Network. Implemented back-propagation algorithm with momentum, auto-encoder network,  dropout during learning, least mean squares algorithm. 

Dataset:

Built a neural network using the MNIST datasets, which provides images of handwritten digits and letters. This is one of the most widely used dataset for evaluating machine learning applications in the image analysis area. Two data files are included: 

1. Image Data File: MNISTnumImages5000.txt is a text file that has data for 5,000 digits, each a grayscale image of size 28 × 28 pixels (i.e., 784 pixels each). Each row of the data file has 784 values representing the intensities of the image for one digit between 0 and 9. The first hundred images are shown in the included file first100.jpg. 
2. Label Data File: MNISTnumLabels5000.txt is a text file with one integer in each row, indicating the correct label of the image in the corresponding row in the image data file. Thus, the first entry ’7’ indicates that the first row of the image data file has data for a handwritten number 7.  
NeuralNetwork_1.1:

1. Wrote a program implementing multi-layer feed-forward neural networks and training them with back- propagation including momentum. The program is be able to handle two hidden layers and any number of hidden neurons, and the user can specify these at run-time. 
2. Randomly chose 4,000 data points from the data files to form a training set, and use the remaining 1,000 data points to form a test set. 
3. Trained a 1-hidden layer neural network to recognize the digits using the training set. I used 10 output neurons – one for each digit – such that the correct neuron is required to produce a 1 and the rest 0. To evaluate performance during training, I used “target values” such as 0.75 and 0.25. Hundreds of epochs were needed for learning, so I considered using stochastic gradient descent, where only a random subset of the 4,000 points is shown to the network in each epoch. The performance of the network in any epoch is measured by the fraction of correctly classified points in that epoch (Hit-Rate).
4. After the network is trained, I tested it on the test set. To evaluate performance on the test data, I used max-threshold approach, where one considers the output correct if the correct output neuron produces the largest output among all 10 output neurons.  
NeuralNetwork_1.2:

1. Trained an auto-encoder network using the same training and testing datasets as in NeuralNetwork_1.1. 
2. Objective is to obtain a good set of features for representing them. 
3. Input to the network is one 28x28 image at a time, and the goal of the network is to produce exactly the same image at the output. 
4. Network is learning a reconstruction task rather than a classification task.
5. Network is trained using back-propagation algorithm using momentum.
6. During training, the value of loss function J2, i.e., value to quantity error, is calculated every 10 epochs.

Report_NeuralNetwork_1:

1. System Description
2. Results - Performance of final network using a Confusion Matrix
3. Features
4. Analysis of Results

NeuralNetwork_2.1: 

1. Light version of ‘Dropout’ by assigning weights to every neuron.
2. Changed the weights only for a randomly chosen subset of inputs and hidden neurons, each chosen with a probability p ~ 0.5.
3. Goal is to make hidden neurons more distinct from each other. 
4. Calculated final loss and error time series. 

NeuralNetwork_2.2:

Using feature detectors found by auto-encoders, and using them as hidden neurons for classification problem. These features are informative enough to be the basis of classification.

Network A-
1. Weights from input to hidden neurons are set to final values of the same weights from the final network in NeuralNetwork_1.2. 
2. During training, each data point is presented to the network and, after the error is calculated, only hidden-to-output weights are adjusted. 

Network B-
1. Weights from input to hidden neurons are set to final values of the same weights from the final network in NeuralNetwork_2.1. 
2. During training, each data point is presented to the network and, after the error is calculated, only hidden-to-output weights are adjusted. 

Report_NeuralNetwork_2:

1. System Description
2. Results - Performance of final network using a Confusion Matrix
3. Features
4. Analysis of Results
