# Charts classification
This project proposes a chart classifier based on AlexNet using fine-tuning to improve the weights of the net.

This project uses the AlexNet model of Caffe framework and the classifier was evaluated using ReVision corpus.

Some details about this project are:
- Deep learning framework: Caffe (<http://caffe.berkeleyvision.org/>)
- Language: Python 2.7
- CNN model: AlexNet model
- Based on GPU
- Optimizer: SGD (Stochastic Gradient Descent: <http://caffe.berkeleyvision.org/tutorial/solver.html>)
- Learning_rate: 1e-4

This project was evaluated using 5-fold cross validation, with 2000 iterations per fold. 
The fold with the best result was the 5th fold (F5); for that reason, the final weights of the models (pretrained and scratch) belong this evaluation. The weights files can be downloaded from the following URL:

<https://drive.google.com/open?id=0BzE_p7Re6WDHM2tNOFlsS0hyUzQ>
