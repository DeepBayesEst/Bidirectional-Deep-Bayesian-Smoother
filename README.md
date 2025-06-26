# Bidirectional-Deep-Bayesian-Smoother
This is the content of the experiment in our paper "Bidirectional Joint State-Memory Deep Bayesian Smoother".   


## Requirements

The code contains the tensorflow implementation of EGBRNN. Different experiments are placed in different paths, each with a corresponding requirement.txt file. To install requirements:

```setup
pip install -r requirements.txt
```

## Data
You can also regenerate or download the raw data. If you want to obtain access to the data we process, please contact us.

For "Aircraft tracking", you can download raw flight records with format "lt6" from the [open resource repository](https://c3.ndc.nasa.gov/dashlink/resources/132/). 
The matlab and python programs used to process the raw data are in the following paths:
~~~~
├── Raw_data_processing
   ├── lt6_to_mat  #  Step 1. Matlab code for processing raw flight records in lt6 format.
   └── mat_to_npy  #  Step 2. Python code for processing data in mat format.
~~~~

For "Vehicle localization", you can download raw records from [XXX dataset](https://robots.engin.umich.edu/nclt/). 

## Training & Testing

The code for each experiment can be find in its corresponding path. We finely commented the code for EGBRNN (tensorflow) in Aircratf tracking, Non-Markov series, and NCLT localization. 
In each experiment (tensorflow):

(1) to train the EGBRNN, please execute:
```Train_DBS.py```

(2) to test the EGBRNN, please execute:
```Test_DBS.py```

For example,

~~~~
├── Filter_and_Smoother_TF
   └── Deep_Bayesian_Filter
         ├── Air_tracking
            ├── EGBRNN
               ├── EGBRNN_train.py
               ├── EGBRNN_test.py
~~~~

<!-- ## Visualisation of Predicted Results
FlightLLM enables highly accurate trajectory prediction in line with flight dynamics.
![Illustrating the prediction result of FlightLLM](Pred_result.png)

## Uncertainty Quantification
FlightLLM can effectively measure the uncertainty of predictions.
![Illustrating the prediction uncertainty of FlightLLM](Uncertainty.png) -->

## Note

If your work involves this code, please pay attention to our our papers: 

[1]Yan S, Liang Y, Zheng L, et al. Explainable Gated Bayesian Recurrent Neural Network for Non-Markov State Estimation[J]. IEEE Transactions on Signal Processing, 2024. (code : https://github.com/DeepBayesEst/EGBRNN_TSP)

[2]Yan S, Liang Y, Zhang H, et al. Explainable Bayesian Recurrent Neural Smoother to Capture Global State Evolutionary Correlations[J]. arXiv preprint arXiv:2406.11163, 2024.

[3]Yan S, Liang Y, Zheng L, et al. Memory-biomimetic deep Bayesian filtering[J]. Information Fusion, 2024, 112: 102580.
