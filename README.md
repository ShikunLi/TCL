# <a href="https://arxiv.org/abs/2203.04199" target="_blank"> Trustable Co-label Learning from Multiple Noisy Annotators </a> - Official PyTorch Code (IEEE TMM 2022)

### Abstract:
Supervised deep learning depends on massive accurately annotated examples, which is usually impractical in many real-world scenarios. A typical alternative is learning from multiple noisy annotators. Numerous earlier works assume that all labels are noisy, while it is usually the case that a few trusted samples with clean labels are available. This raises the following important question: how can we effectively use a small amount of trusted data to facilitate robust classifier learning from multiple annotators? This paper proposes a data-efficient approach, called Trustable Co-label Learning (TCL), to learn deep classifiers from multiple noisy annotators when a small set of trusted data is available. This approach follows the coupled-view learning manner, which jointly learns the data classifier and the label aggregator. It effectively uses trusted data as a guide to generate trustable soft labels (termed co-labels). A co-label learning can then be performed by alternately reannotating the pseudo labels and refining the classifiers. In addition, we further improve TCL for a special complete data case, where each instance is labeled by all annotators and the label aggregator is represented by multilayer neural networks to enhance model capacity. Extensive experiments on synthetic and real datasets clearly demonstrate the effectiveness and robustness of the proposed approach.

### Requirements:
* Python 3.8.10
* Pytorch 1.8.0 (torchvision 0.9.0)
* Numpy 1.19.5

### Download Datasets
Download the training dataset with annotations from the corresponding link and put the prepared data into ./data folder 
[LabelMe-AMT](http://fprodrigues.com//deep_LabelMe.tar.gz)

### Running the code:
To run the code use the provided script in LabelMe-AMT folders. 

### Citation:
If you find the code useful in your research, please consider citing our paper:

```
 @article{Li2022TCL,
  title = {Trustable Co-label Learning from Multiple Noisy Annotators},
  authors = {Shikun Li and Tongliang Liu and Jiyong Tan and Dan Zeng and Shiming Ge},
  year={2022},
  booktitle ={IEEE TMM},
 } 
```

Note: Our implementation uses parts of "Deep Learning from Crowds" https://github.com/fmpr/CrowdLayer implementations.