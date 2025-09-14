# thermal-fall-benchmark
Code and resources for reproducibility of the IEEE TAI paper on thermal fall detection using the TF-66 and TSF datasets. Includes M2 model implementation, standardized 80:20 splits, and dataset access links.

The file "TSF_Dataset_Split.xlsx" outlines the train / validation split for the TSF dataset, with the yellow highlighted lines representing the validation portion of the dataset. 

The file "Data_Generator.py" includes the data generator used for training that other researchers are encouraged to adopt - this also includes the subset toggle for each training on specfic subsets. 

The file "Metrics_and_Attentions.py" includes class definitions for metrics and attention modules used in this work.

The file "Model.py" includes the best performing model - M2 - from the published work.

The TF-66 dataset used in this manuscript (including all information about how to use it, caching instructions, and train/validation split) can be accessed at the following github link: https://github.com/Christopher-Silver/TF-66.

Licensed under Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0). See LICENSE file for details.
