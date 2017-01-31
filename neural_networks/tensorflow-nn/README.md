# ucsp-mcs-ia-tensorflow-nn
Creating a neural network using Tensorflow and Python.

This code was tested with the following files:
- Training data: car_train.csv
- Test data: car_train.csv
- Data to predict: car_pred.csv

Furthermore, in the code we have three types of classifiers:
- "wide": ```tf.contrib.learn.LinearClassifier```
- "deep": ```tf.contrib.learn.DNNClassifier```
- "wide_n_deep": ```tf.contrib.learn.DNNLinearCombinedClassifier```

The model saved on car_model folder was generated using the following parameters:
- model_type: "deep"
- train_steps: 500
- hidden_units: two hidden layers, the first with 100 units and the second with 50 units
- optimizer: ```tf.train.GradientDescentOptimizer```
- learning_rate: 0.05

This model has equal prediction results than Naive Bayes method, we show the details:

```
--- Tensorflow results ---
Accuracy: 0.961137
MAP_CLASSES = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
Predictions: [0, 2, 0, 0]

--- Naive Bayes results ---
0 -> low - low - 4 - 2 - big - high - CLS: unacc
1 -> low - low - 5more - 4 - med - med - CLS: good
2 -> low - low - 5more - more - big - low - CLS: unacc
3 -> high - vhigh - 3 - 4 - big - med - CLS: unacc
```

