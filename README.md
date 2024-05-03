# Evaluation code for the NeurIPS 2023 Machine Unlearning competition


This is the code used to score submissions during the NeurIPS 2023 Machine
Unlearning competition.

## Installation

To install dependencies:

  `pip install -r requirements.txt`

This code also requires to have the CASIA-SURF dataset. To get this dataset,
go to the dataset webpage:

  https://sites.google.com/corp/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020

Download the dataset license agreement from that webpage and send it to Jun Wan
(jun.wan@ia.ac.cn). Once this is done, you should receive a link to the dataset.

Once you receive the dataset link, download the files `Age_Gender.txt` and
`CASIA_SURF_live.zip`, and unzip the latter.


## Usage

The script launch.py will launch the evaluation metric and can be run as

  xmanager launch.py

This will call `main.py` with the specified flags for the data directory,
checkpoint directory and output directory.

The code currently evaluates a particular unlearning function (unlearning by
simple fine-tuning) which is specified in main.py (see function `do_unlearning`
). The user can replace the contents of `do_unlearning` with another unlearning
algorithm.

Overall, the code is structured as follows: the given unlearning algorithm will
be ran for `_NUM_MODELS` times (512 by default) and, similarly, the oracle
retrain-from-scratch algorithm will also be ran for that number of times. The
retrain-from-scratch checkpoints are saved in the checkpoint directory, so that
the second time that the code is ran they will be reloaded, to save compute.
The forgetting quality score is then computed from the confidences of each
forget example obtained by the unlearned and retrained models. That score is
finally adjusted to take utility (accuracy on the retain and test sets) under
consideration, yielding a final score.

Please refer to this technical report for the mathematical definition of
unlearning that this metric is based on, as well as a more detailed description
of the evaluation metric:
https://unlearning-challenge.github.io/assets/data/Machine_Unlearning_Metric.pdf


## Citing this work

```
@misc{neurips-2023-machine-unlearning,
    author = {Eleni Triantafillou, Fabian Pedregosa, Jamie Hayes, Peter Kairouz, Isabelle Guyon, Meghdad Kurmanji, Gintare Karolina Dziugaite, Peter Triantafillou, Kairan Zhao, Lisheng Sun Hosoya, Julio C. S. Jacques Junior, Vincent Dumoulin, Ioannis Mitliagkas, Sergio Escalera, Jun Wan, Sohier Dane, Maggie Demkin, Walter Reade},
    title = {NeurIPS 2023 - Machine Unlearning},
    publisher = {Kaggle},
    year = {2023},
    url = {https://kaggle.com/competitions/neurips-2023-machine-unlearning}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
