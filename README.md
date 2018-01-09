# Getting Started

This project uses statistical analysis tools to predict county election results.
Here we describe how to setup the development environment and discuss project structure.
See NOTES.md for more information about model selection and other information.

# Prerequisites

1. Install vagrant and VirtualBox on the host machine. Ensure that they are included in the PATH environment.
	
    + `Vagrant`: [https://www.vagrantup.com/](https://www.vagrantup.com/)
    + `Virtualbox`: [https://www.virtualbox.org/wiki/Downloads](https://www.virtualbox.org/wiki/Downloads)

2. Unzip and navigate to the root of the project directory.
3. Ensure port '8888' is not being used by another process on the host machine.

# Installation

1. To set up the environment and start running the project run:

    > `vagrant up`
    > `vagrant ssh`

2. Navigate to 'http://localhost:8888' and ensure that Jupyter server is running.

# Quick Start

1. In the browser navigate to Analysis.ipynb. This file will give some intuition to model selection and procedure.
2. Then view Model.ipynb where different models are tested and optimized.
3. In the vagrant terminal session run:

    > `python3 build_model.py`
    > `python3 make_predictions.py`

4. The file predictions.csv in the root directory contains predicted class labels of the testing dataset './Data/test_potus_by_county.csv'.

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
	http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
