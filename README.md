### Sequential Recommendation with Collaborative Explanation via Mutual Information Maximization (SIGIR 24)

This repository provides code for the paper "Sequential Recommendation with Collaborative Explanation via Mutual Information Maximization".

### Environment
See requirements.txt

### Data Preparation
We use EXTRA data set from https://github.com/lileipisces/EXTRA.

You can also download the processed data via: [https://drive.google.com/drive/folders/10lVVWCLsRL2SOibG_s85917IxH9UVnqh](https://drive.google.com/drive/folders/11cAMwj7ZpmDXhKSDs4LAnmDqvScyIp5m?usp=sharing)
    



### Example

For tuning hyperparamters: 

    python main.py --dataset Amazon --test_mode 0

For training and inference: 

    python main.py --dataset Amazon --test_mode 1

If you consider using this code or its derivatives, please consider citing.
