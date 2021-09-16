# SSD: A Unified Framework for Self-Supervised Outlier Detection [ICLR 2021]

Pdf: https://openreview.net/forum?id=v5gjXpmR8J

Code for practice ICLR 2021 paper on outlier detection, titled SSD, without requiring class labels of in-distribution training data.


## Getting started

Prepare Dependency

`pip install -r requirement.txt`

## Sample Results

```
In-data = cifar10, OOD = cifar100, Clusters = 1, FPR95 = 0.5078, AUROC = 0.9063240349999999, AUPR = 0.8919609510086947
In-data = cifar10, OOD = svhn, Clusters = 1, FPR95 = 0.020666871542716656, AUROC = 0.9962383988936693, AUPR = 0.9985624119973668
```
## Reference

If you find this work helpful, consider citing it. 

```
@inproceedings{sehwag2021ssd,
  title={SSD:  A Unified Framework for Self-Supervised Outlier Detection},
  author={Vikash Sehwag and Mung Chiang and Prateek Mittal},
 booktitle={International Conference on Learning Representations},
 year={2021},
 url={https://openreview.net/forum?id=v5gjXpmR8J}
}
```
