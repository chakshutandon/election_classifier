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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from datetime import datetime
	
def getFeatureLabelColumns(dataframe, CLASS_LABELS_COLUMN):
    """Split dataframe into features and labels. CLASS_LABELS_COLUMN is the column header for the label column."""
    features = dataframe.loc[:, dataframe.columns != CLASS_LABELS_COLUMN]
    labels = dataframe.loc[:, CLASS_LABELS_COLUMN]
    return features, labels

def scaleUnitMean(inputMatrix, scaler_=None):
    """Scale input matrix to zero mean and unit variance. Pass in instance of sklearn.preprocessing.StandardScaler() to retain internal state."""
    if scaler_ is None:
    	scaler_ = StandardScaler()
    return scaler_.fit_transform(inputMatrix), scaler_

def labelEncode(inputMatrix):
    """Convert class labels to binary representations for training."""
    le_ = LabelEncoder()
    return le_.fit_transform(inputMatrix), le_

def labelDecode(inputMatrix, le_):
    """Convert binary representations of class labels to string representations. Pass in instance of sklearn.preprocessing.LabelEncoder() to retain internal state from training."""
    return le_.inverse_transform(inputMatrix)

def computePCA(inputMatrix, n_components=None):
    """Compute Principle Component Analysis (PCA) on feature space. n_components specifies the number of dimensions in the transformed basis to keep."""
    pca_ = PCA(n_components)
    pca_.fit(inputMatrix)
    return pca_

def applyPCA(inputMatrix, pca_):
    """Apply Principle Component Analysis (PCA) on feature space. Pass in instance of sklearn.decomposition.PCA()."""
    return pca_.transform(inputMatrix)

def saveObjectToFilesystem(obj, filename):
    """Save objects to binary files on filesystem."""
    joblib.dump(obj, filename)

def getObjectFromFilesystem(filename):
    """Retrieve objects from binary files on the filesystem."""
    return joblib.load(filename)

def getPredictedLabels(model, features):
    """Run inference on trained model to get predicted class labels."""
    return model.predict(features)

def buildPerformanceString(metrics, confMatrix):
    """Build string representation of model metrics to print and save to filesystem."""
    result = "----------------------------------------------\n"
    result += "Performance ({:%B %d, %Y %I:%M:%S %p}):\n".format(datetime.now())
    for metric, scores in metrics.items():
        result += "\t{}: {:.2} (+/- {:.2})\n".format(metric, scores.mean(), scores.std())
    result += "\nConfusion Matrix:\n"
    result += np.array_str(confMatrix) + "\n"
    return result

def saveStringToFilesystem(filename, string):
    """Write string to file."""
    with open(filename, 'a') as f:
        f.write(string)
