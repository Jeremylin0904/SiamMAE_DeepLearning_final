# Final Project of Deep Learning - Advanced Course

In this project, we aim to reproduce the Siamese Masked AutoEncoder (SiamMAE) proposed in [Siamese Masked Autoencoders](https://arxiv.org/abs/2305.14344) using the PyTorch framework.
The dataset we will be pretraining our model will be UCF-101 [UCF101: A Dataset of 101 Human Actions
Classes From Videos in The Wild](https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf). The representation learned by the model will be evaluated on an object segmentation task on DAVIS-2017 [The 2017 DAVIS Challenge on Video Object Segmentation](https://arxiv.org/abs/1704.00675).

This project was part of the course [DD2412 Deep Learning, Advanced Course](https://www.kth.se/student/kurser/kurs/DD2412?l=en) at KTH.

Please refer to the notebook example to see how to run the experiments. We could not upload our trained model to the Github due to storage limiations.

## Results
Example of results from object segmentation, the first video was an easy example whereas the second was an harder one. For more details, refer to our project report.
![davis_bear](https://github.com/Jeremylin0904/DeepLearning_final/assets/117983459/0c434532-417e-4509-817a-eba42a35e3e1)
![davis_bike](https://github.com/Jeremylin0904/DeepLearning_final/assets/117983459/c4ebc99d-a696-4d11-9fa5-4faa339242ec)
