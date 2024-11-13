![image](https://github.com/user-attachments/assets/1c16e490-4276-45f7-9bac-2cbc69eea99d)# OSDD
The official implementation of the IEEE TIFS paper "Fine-grained Open-set Deepfake Detection via Unsupervised Domain Adaptation"
[An initial cleaning up of the code for quick reference. We will update this repository ASAP.]

# Method overview
The framework of our unsupervised domain adaptation method for fine-grained open-set deepfake detection. First, use labeled data from the source domain to train a feature extraction network Encoder. Then, use Encoder to extract features FT from unlabeled data XT in the target domain. The extracted features FT of the target domain are then used for clustering the unlabeled target domain images XT using Network Memorization-based Adaptive Clustering (NMAC) to convert the previously unclassified source domain features into labeled source domain features. Next, through the Target Domain Pseudo Label Generator (PLG), the adaptive clustering results of the target domain are associated with the known depth-forging methods of the source domain to obtain the best clustering result for the target domain data, resulting in $Y*_T$ . Finally, the pre-trained M in the first step is retrained using labeled data from the source domain and target domain data optimized by PLG to obtain the desired forging method through Model Retraining (MR).

#  Citation

# Contact
For technique issues with this work, you may contact：zhouxinye21[at]mails.ucas.ac.cn, or zxy072601[at]163.com
