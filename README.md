# Project 1 CIFAR-10 Classification

by zxf

## ==The following is the correspondence between tasks and source code==



### Please  place  the .pt  data  file  in the  project  directory  before  running

# Code Architecture

## Required Tasks:

1. Implement a two-layer MLP with hidden dims 128 
    main.py			model.py

2. change the number of layers of the MLP and report your discovery
     main_0.py		model_0.py

3. Implement a simple convolution network according to the architecture given in the following page
    main.py			model.py

4. Hyper-parameter tuning
    main_1.py        model.py
    main_1_\*.py        model.py
    
5. Add Dropout and discover its effect
    main_2.py        model_2.py

6. Use multiple metrics (Acc, F1…) to evaluate
    main_3.py        model.py

7. Compare different networks’ capacity

8. Underfitting/overfitting

## Exploratory Tasks

1. Add data-augmentation and verify its effectiveness
     main_ex_01.py               model.py

1. Add batch norm in the convolutional network and verify its effectiveness
     main_ex_02.py               model_ex_02.py

2. Implement a self-attention layer and insert it to somewhere in the convolution network
     main_ex_03.py               model_ex_03.py

     main_ex_03\_\*.py               model_ex\_\*.py（Hyper-parameter tuning）

