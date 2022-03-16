# More Metaphor Detection

Adapting [**IlliniMet: Illinois System for Metaphor Detection**](https://github.com/HongyuGong/MetaphorDetectionSharedTask):

    Hongyu Gong, Kshitij Gupta, Akriti Jain and Suma Bhat 
    "IlliniMet: Illinois System for Metaphor Detection with Contextual and Linguistic Information",
    in Proceedings of the Second Workshop on Figurative Language Processing 2020 (pp. 146--153).
    
    https://aclanthology.org/2020.figlang-1.21.pdf

We use the RoBERTa model from Gong et. al. to perform metaphor detection on MOH-X, TroFi, and VUA datasets. Our goal is to compare RoBERTa performance to BiLSTM.
### Project Overview

Language Requirements: Python 3.9

Library Requirements:

1. [Transformers by Huggingface](https://github.com/huggingface/transformers)

### Data
We are using the processed data from our project to replicate [gao et. al.](https://github.com/cmsc-25700/metaphor-detection)
1. MOH-X
2. TroFi
3. VUA

The gao et. al. versions of the data are in [resources](resources/metaphor-in-context/data).
The reformatted versions for the BERT model are in [data](data).

### Models
We use the RoBERTa model with and without POS. (For MOH-X and TroFI POS tags are obtained from spaCy).
See [shell-scripts](shell-scripts) for code to train and predict.
