# Bidirectional-Deep-Bayesian-Smoother
This is the content of the code in our paper "Bidirectional Joint State-Memory Deep Bayesian Smoother".   


## Requirements

The code contains the tensorflow implementation of EGBRNN. Different experiments are placed in different paths, each with a corresponding requirement.txt file. To install requirements:

```setup
pip install -r requirements.txt
```

## Data
You can also regenerate or download the raw data. 

For "Aircraft tracking", you can download raw flight records with format "lt6" from the [open resource repository](https://c3.ndc.nasa.gov/dashlink/resources/132/). 
The matlab and python programs used to process the raw data are in the following paths:
~~~~
├── Raw_data_processing
   ├── lt6_to_mat  #  Step 1. Matlab code for processing raw flight records in lt6 format.
   └── mat_to_npy  #  Step 2. Python code for processing data in mat format.
~~~~

For "Vehicle localization", you can download the dataset by following the references provided in our paper.

In addition, we also provide examples of the preprocessed data.

## Training & Testing

We have provided annotations for the code related to aircraft trajectory smoothing and vehicle localization, so you can easily locate and run them based on the directory structure.

## Note

If your work involves this code, please pay attention to our our papers: 

[1]Yan S, Liang Y, Zheng L, et al. Explainable Gated Bayesian Recurrent Neural Network for Non-Markov State Estimation[J]. IEEE Transactions on Signal Processing, 2024. (code : https://github.com/DeepBayesEst/EGBRNN_TSP)

[2]Yan S, Liang Y, Zhang H, et al. Explainable Bayesian Recurrent Neural Smoother to Capture Global State Evolutionary Correlations[J]. arXiv preprint arXiv:2406.11163, 2024.

[3]Yan S, Liang Y, Zheng L, et al. Memory-biomimetic deep Bayesian filtering[J]. Information Fusion, 2024, 112: 102580.
