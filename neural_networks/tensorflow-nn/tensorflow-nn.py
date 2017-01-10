from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import itertools

# Data sets config
CAR_TRAIN = "car_train.csv"
CAR_TEST = "car_train.csv"
CAR_PRED = "car_pred.csv"
NUM_FEATURES = 6
NUM_CLASSES = 4

model_dir = "."
model_type = "deep" # {"deep", "wide_n_deep", "wide"}
train_steps = 500
hidden_units = [100, 50]

# Config of data
MAP_CLASSES = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "cls"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
CONTINUOUS_COLUMNS = []


def build_estimator(model_dir):
    """Build an estimator."""
    buying = tf.contrib.layers.sparse_column_with_keys(column_name="buying", keys=["vhigh", "high", "med", "low"])
    maint = tf.contrib.layers.sparse_column_with_keys(column_name="maint", keys=["vhigh", "high", "med", "low"])
    doors = tf.contrib.layers.sparse_column_with_keys(column_name="doors", keys=["2", "3", "4", "5more"])
    persons = tf.contrib.layers.sparse_column_with_keys(column_name="persons", keys=["2", "4", "more"])
    lugboot = tf.contrib.layers.sparse_column_with_keys(column_name="lug_boot", keys=["small", "med", "big"])
    safety = tf.contrib.layers.sparse_column_with_keys(column_name="safety", keys=["low", "med", "high"])

    wide_columns = [buying, maint, doors, persons, lugboot, safety]
    deep_columns = [
        tf.contrib.layers.embedding_column(buying, dimension=8),
        tf.contrib.layers.embedding_column(maint, dimension=8),
        tf.contrib.layers.embedding_column(doors, dimension=8),
        tf.contrib.layers.embedding_column(persons, dimension=8),
        tf.contrib.layers.embedding_column(lugboot, dimension=8),
        tf.contrib.layers.embedding_column(safety, dimension=8),
    ]
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(
            model_dir=model_dir, 
            feature_columns=wide_columns,
            n_classes=NUM_CLASSES,
            optimizer=optimizer)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            n_classes=NUM_CLASSES,
            optimizer=optimizer)
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            n_classes=NUM_CLASSES)
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values, 
            shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval():
    """Train and evaluate the model."""
    df_train = pd.read_csv(
        tf.gfile.Open(CAR_TRAIN),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(CAR_TEST),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")

    # Samples to predict classes
    df_pred = pd.read_csv(
        tf.gfile.Open(CAR_PRED),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")

    df_train[LABEL_COLUMN] = (df_train["cls"].apply(lambda x: MAP_CLASSES[x])).astype(int)
    df_test[LABEL_COLUMN] = (df_test["cls"].apply(lambda x: MAP_CLASSES[x])).astype(int)
    df_pred[LABEL_COLUMN] = (df_pred["cls"].apply(lambda x: 0)).astype(int)
    
    m = build_estimator(model_dir)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    print('\nAccuracy: {0:f}'.format(results["accuracy"]))

    y = m.predict(input_fn=lambda: input_fn(df_pred))
    predictions = list(itertools.islice(y, 4))
    print ("Predictions: {}".format(str(predictions)))


def main(_):
    train_and_eval()


if __name__ == "__main__":
    tf.app.run()
