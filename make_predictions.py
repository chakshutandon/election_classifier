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

import configparser
import utils

config = configparser.ConfigParser()
config.read('config.ini')
config = config['DEFAULT']

def main():
    print("Reading trained model...", end='')
    try:
        model = utils.getObjectFromFilesystem('model')                                                              # Read model from binary file
    except FileNotFoundError:
        print("Model binary does not exist! Have you run build_model.py?")
    print("Done")

    print("Reading testing data...", end='')
    try:
        testing_data_ = pd.read_csv(config['TestingCSVPath'])
    except FileNotFoundError:
        print("Could not find {}!".format(config['TestingCSVPath']))
    print("Done")

    print("Transforming features...", end='')
    try:
        transformations = utils.getObjectFromFilesystem('transformations')                                          # Read transformations binary to get internal states of scale, encode, and PCA during training
    except FileNotFoundError:
        print("Transformations binary does not exist! Have you run build_model.py?")
    
    features, scaler_ = utils.scaleUnitMean(testing_data_, transformations['scaler'])                               # Apply scaling operation to features
    pca_transformed_features = utils.applyPCA(features, transformations['pca'])                                     # Apply PCA transformation to features
    print("Done")

    print("Predicting Class Labels...", end='')
    predictedClasses = utils.getPredictedLabels(model, pca_transformed_features)                                    # Run inference on model
    predictedLabels = utils.labelDecode(predictedClasses, transformations['encoder'])                               # Decode binary class predictions to original labels
    print("Done")

    np.savetxt("predictions.csv", predictedLabels, delimiter=',', fmt="%s", header='Winner', comments='')           # Save predictions to predictions.csv
    print("Class labels saved to predictions.csv")

if __name__ == "__main__": 
    main()
