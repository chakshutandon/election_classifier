# Author: Chakshu Tandon <chakshutandon@gmail.com>
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import configparser
import utils

config = configparser.ConfigParser()
config.read('config.ini')
config = config['DEFAULT']

def computeMetrics(model, features, labels):
    """
    Compute accuracy, f1 score, area under ROC curve, and confusion matrix for trained model.

    Args:
        model: Trained sklearn classifier.
        features: Matrix of size (row*col).
        labels: Matrix of size (row*n_classes) containing class labels.
    
    Returns:
        metrics: Python dictionary with accuracy, f1, and roc_auc.
        confMatrix: Confusion matrix between predicted and true labels.
    """
    metrics = {}
    for metric in ['accuracy','f1', 'roc_auc']:
        cvScores = cross_val_score(model, features, labels, scoring=metric, cv=int(config['n_KFolds']))
        metrics[metric] = cvScores

    predictedLabels = utils.getPredictedLabels(model, features)
    confMatrix = confusion_matrix(labels, predictedLabels)

    return metrics, confMatrix

def buildRFModel(features, labels):
    """
    Build a random forrest classifier and fit using training data.

    Args:
        features: Matrix of size (row*col).
        labels: Matrix of size (row*n_classes) containing class labels.
    
    Returns:
        RandomForestClassifier: A trained instance of the RandomForestClassifier .
    """
    return RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1).fit(features, labels)

def main():
    print("Reading training data...", end='')
    try:
        training_data_ = pd.read_csv(config['TrainingCSVPath'])
    except FileNotFoundError:
        print("Could not find {}!".format(config['TrainingCSVPath']))
    features_, labels_ = utils.getFeatureLabelColumns(training_data_, config['ClassLabelsColumn'])                  # Split dataframe by feature and label columns to train
    print("Done")

    print("Transforming features...", end='')
    features, scaler_ = utils.scaleUnitMean(features_)                                                              # Scale data to zero mean and unit variance
    labels, encoder_ = utils.labelEncode(labels_)                                                                   # Convert class labels to binary values
    pca_ = utils.computePCA(features, n_components=int(config['n_PCAComponents']))                                  # Compute Principle Component Analysis (PCA) on features
    pca_transformed_features = utils.applyPCA(features, pca_)                                                       # Apply PCA

    transformations_ = {'scaler':scaler_, 'encoder':encoder_, 'pca':pca_}
    utils.saveObjectToFilesystem(transformations_, 'transformations')                                               # Save scale, encode, and PCA states to apply during inference
    print("Done")

    print("Building model (this may take a while)...", end='')
    model = buildRFModel(pca_transformed_features, labels)
    utils.saveObjectToFilesystem(model, 'model')                                                                    # Save model to binary file
    print("Done")

    print("Training (this may take a while)...", end='')
    metrics, confMatrix = computeMetrics(model, pca_transformed_features, labels)                                   # Compute accuracy, f1 score, area under ROC curve, and confusion matrix
    print("Done")

    performance = utils.buildPerformanceString(metrics, confMatrix)                                                 # Build string representation of performance stats
    print("\n{}".format(performance))
    utils.saveStringToFilesystem('performance.txt', performance)

    print("\nTrained model saved to ./model binary.")

if __name__ == "__main__": 
    main()
