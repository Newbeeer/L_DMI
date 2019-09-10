# Introduction

Implementation for NeurIPS 2019 paper : <img src="https://github.com/Newbeeer/L_DMI/blob/master/title.png" width="450px" />

paper link: https://arxiv.org/abs/1909.03388



### Fashion MNIST dataset

- To run experiments of *Fashion MNIST* dataset in `fashion` directory:

```shell
python3 fashion.py --r noisy_amount --s seed --c case_num --device device_num

noise_amount: the amount of noise amount r of label flipping. (0 <= r <= 1)
seed: random seed
case_num :  1: class-independent; 2: class-dependent (a); 3: class-dependent (b)
device_num: GPU number

```



### CIFAR-10 dataset:

- To run experiments of *CIFAR10* dataset in `CIFAR-10` directory, pleases run all the baseline in the following order: 

  ```shell
  python3 CE.py --r noise_amount --s seed --device device_num
  python3 FW.py --r noise_amount --s seed --device device_num
  python3 GCE.py --r noise_amount --s seed --device device_num
  python3 LCCN.py --r noise_amount --s seed --device device_num
  python3 DMI.py --r noise_amount --s seed --device device_num
  
  noise_amount: the amount of noise amount r of label flipping. (0 <= r <= 1)
  seed: random seed
  device_num: GPU number
  ```

  

### Dog & Cat datasete:

- To run experiments of *Dog vs. Cats* dataset in `dogcat` directory, pleases run all the baseline in the following order: 

  ```shell
  python3 CE.py --r noise_amount --s seed --device device_num
  python3 FW.py --r noise_amount --s seed --device device_num
  python3 GCE.py --r noise_amount --s seed --device device_num
  python3 LCCN.py --r noise_amount --s seed --device device_num
  python3 DMI.py --r noise_amount --s seed --device device_num
  
  noise_amount: the amount of noise amount r of label flipping. (0 <= r <= 1)
  seed: random seed
  device_num: GPU number
  ```

  

### Clothing1M dataset

- To run experiments of *Clothing1M* dataset in `clothing` directory, pleases run all the baseline in the following order: 

  ```shell
  python3 main.py --device device_num
  
  device_num: GPU number
  ```



### TODO:

Combine all the APIs to dataset into one file.





