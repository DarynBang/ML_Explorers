<!-- PAPER TITLE -->

# Machine Learning Assignment 1

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#enviroments">Enviroments</a></li>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#embedding-method">Embedding method</a></li>
    <li><a href="#machine-learning-algorithm">Machine learning algorithm </a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Authors

<ol>
    <li><a href="">Tieu Tri Bang       - 2252079</a></li>
    <li><a href="">Nguyen Van Hieu     - 2153345</a></li>
    <li><a href="">Pham Huy Thien Phuc - 2053346</a></li>
</ol>

## Enviroments
All experiments run successfully on Google Colab with:

- Python version: 3.11.11
- PyTorch version: 2.5.1+cu124

## Introduction
The classification of 3D medical images presents unique challenges and opportunities in the field of medical imaging and machine learning. In this study, we leverage the MedMNIST [[1]](#1) dataset, a standardized benchmark for medical imaging tasks, to develop and evaluate classifiers for 3D image data. We explore three distinct embedding approaches: a flatten-based method, a 2D Convolutional Neural Network (CNN), and a 3D CNN, each designed to capture spatial features at varying levels of complexity. Results demonstrate that the 3D CNN outperforms simpler embeddings in capturing intricate volumetric patterns, though trade-offs in computational cost are notable. This work provides insights into the suitability of these approaches for 3D medical image classification and establishes a foundation for future advancements in automated diagnostic systems.
## Datasets
<img src="assets/medmnist++.png" width="1000">

In this study, we utilize six 3D datasets from MedMNIST—OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, and SynapseMNIST3D—each preprocessed to a standardized 28x28x28 resolution, to investigate multi-class classification of anatomical and pathological features.


## Embedding 
### Method
<img src="assets/flatten.png" width="400">

**Flatten**: In this method, each 3D volume is unraveled into a 1D vector of length 21,952 (28 × 28 × 28). While computationally simple and requiring minimal architectural complexity, this approach discards all spatial relationships inherent in the volumetric data, treating each voxel as an independent feature. As a result, it relies entirely on the subsequent dense layers to learn any patterns or dependencies, making it less suited for capturing the rich 3D structural information present in medical imaging tasks.

<img src="assets/2d.png" width="700">

**2D-CNN**: The 2D-CNN embedding adapts a conventional 2D convolutional approach to handle the 3D 28x28x28 volumes by treating the depth dimension as a channel-like feature. Specifically, each 3D volume is processed as a stack of 28 two-dimensional 28x28 slices, where the depth (z-axis) is interpreted as 28 input channels. A 2D-CNN architecture, consisting of convolutional layers, pooling layers, and fully connected layers, is then applied to extract spatial features from each slice. This method leverages well-established 2D convolutional operations to capture patterns within individual planes of the volume, such as edges or textures, but it does not explicitly model correlations across the depth dimension. While more sophisticated than the flatten approach, the 2D-CNN may struggle to fully represent the volumetric relationships critical for 3D medical image classification, potentially limiting its performance on tasks requiring holistic 3D understanding.

<img src="assets/3d.png" width="700">

**3D-CNN**: The 3D-CNN embedding is designed to fully exploit the volumetric nature of the 28x28x28 MedMNIST datasets by applying 3D convolutional operations. In this approach, the input volume is processed directly as a 3D tensor, with convolutional kernels extending across all three dimensions (x, y, z). The architecture typically includes multiple 3D convolutional layers followed by pooling layers to reduce spatial dimensions, culminating in fully connected layers for classification. This method captures spatial dependencies and contextual information across the entire volume, making it particularly well-suited for identifying complex 3D structures, such as organ boundaries, nodule shapes, or vascular networks. However, the increased representational power comes at the cost of higher computational complexity and memory requirements compared to the flatten and 2D-CNN methods.

### Feature Analysis

Note: 'nodulemnist3d' is the class 0, 'vesselmnist3d' is the class 1, 'synapsemnist3d' is the class 2, 'adrenalmnist3d' is the class 3, 'fracturemnist3d' is the class 4, 'organmnist3d' is the class 5 in all the images in this section

#### Flatten:

<img src="assets/reduce_dimension_flatten.png" width="1200">
The PCA visualizations of the train, validation, and test sets show remarkably similar distributions, with clusters of points, particularly for Classes 0 and 2, appearing quite consistent across the three datasets. However, a significant drawback is the lack of separation between classes, as there is substantial overlapping, making it challenging to distinguish between them effectively. In contrast, the t-SNE visualizations reveal distinctly different distributions across the train, validation, and test sets, with the training data showing tighter, more separated clusters compared to the validation and test sets. This discrepancy suggests that while t-SNE better captures the local structure and separability within the training data, the model fitted on this distribution struggles to generalize to the validation and test sets, indicating potential issues with overfitting or insufficient representation of the data's variability across all stages.

#### 2D-CNN:
**Initialization weight:**

<img src="assets/reduce_dimension_2d_init.png" width="1200">

The PCA visualizations of the 2D CNN initialized weight embeddings (before training) for the train, validation, and test sets demonstrate a consistent distribution across all three datasets, suggesting that the initialization preserves a uniform structure before any training occurs. Compared to the flatten embeddings, these 2D CNN embeddings appear to offer an advantage, as they can capture the spatial relationships of 2D dimensions such as width and height, leading to potentially richer representations of the data. Furthermore, the distributions in these PCA plots show improved separation among classes compared to the flatten embeddings; for instance, Class 2 forms a distinct, tight cluster, indicating better class separability. This enhanced structure highlights the benefit of leveraging 2D CNN architectures, which may better preserve meaningful patterns in the data, even in the initial, untrained state.

**Pretrained weight:**

<img src="assets/reduce_dimension_2d_pretrained.png" width="1200">
The PCA visualizations of the 2D CNN pretrained weight embeddings, which have been trained on the training data and validated on the validation data, exhibit a consistent distribution across the train, validation, and test sets, indicating that the pretrained model maintains a stable structure across all datasets. This consistency is an improvement over the flatten embeddings, as the 2D CNN architecture effectively captures the spatial relationships of 2D dimensions such as width and height, resulting in more robust representations of the data. Additionally, the distributions in these PCA plots show better separation among classes compared to the flatten embeddings, with the train, validation, and test sets displaying similar patterns and shapes, such as the distinct linear arrangement of points across classes. This similarity in patterns across datasets suggests that the pretrained 2D CNN model generalizes well, preserving meaningful class distinctions and spatial information effectively.

#### 3D-CNN:
**Initialization weight:**

<img src="assets/reduce_dimension_3d_init.png" width="1200">
The PCA and t-SNE visualizations of the 3D CNN initialized weight embeddings (before training) reveal a relatively consistent distribution across the train, validation, and test sets, indicating that the initialization maintains a similar structure across all three datasets. However, despite this uniformity, there is a lack of clear separation among the classes, as evidenced by significant overlap, particularly between Class 1 and Class 3, where points from these classes intermingle in both the PCA and t-SNE plots. This overlapping suggests that the initial embeddings, while structurally similar across datasets, do not yet capture distinct features necessary for effective class discrimination. The similarity in distribution highlights the stability of the 3D CNN’s initialization, but the lack of separation indicates that further training or feature engineering may be required to enhance class distinguishability.

**Pretrained weight:**

<img src="assets/reduce_dimension_3d_pretrained.png" width="1200">
The PCA and t-SNE visualizations of the 3D CNN pretrained weight embeddings, which have been trained on the training data and validated on the validation data, consistently demonstrate clear clustering and a similar distribution across the train, validation, and test sets. This uniformity highlights the model’s ability to effectively capture and preserve the spatial information of the three-dimensional data, leveraging the 3D CNN’s capacity to understand the depth, width, and height dimensions. In both PCA and t-SNE plots, classes form distinct clusters—such as the tight grouping of blue (Class 1) and green (Class 2) points—indicating improved separability compared to earlier stages or other architectures. The consistent patterns across datasets underscore the pretrained 3D CNN’s robustness in generalizing spatial relationships, making it well-suited for handling complex 3D data structures.

### Computational Efficiency Analysis
This document provides a computational comparison between Flattening, 2D Convolutional Neural Networks (2D CNN), and 3D Convolutional Neural Networks (3D CNN) in terms of parameter count, memory consumption, and computational complexity.

#### 1. Flatten
Flattening transforms a multi-dimensional input (e.g., images or volumes) into a 1D vector.

**Advantages:**
- Simple and computationally efficient.
- Requires fewer parameters than CNNs.
- Suitable for simple tasks with small datasets.

**Disadvantages:**
- Ignores spatial information (no local feature extraction).
- Performs poorly on complex structured data.

**Computational Complexity:**
- $$O(n)$$ (linear transformation)

####  2. 2D CNN (Convolutional Neural Network)
2D CNNs process 2D images using convolutional layers with kernels (filters) that slide spatially across width and height.

**Advantages:**
- Captures spatial hierarchies in data.
- Reduces parameter count compared to fully connected networks.
- Efficient for image-based tasks.

**Disadvantages:**
- Limited to processing 2D information; loses depth-wise relationships.
- Not suitable for volumetric data (e.g., medical imaging, video frames).


**Computational Complexity:**
For a single convolution layer with input size $$(H , W , C_{in})$$, kernel size $$(K , K , C_{in} , C_{out})$$, and output size $$(H' , W' , C_{out})$$:
- $$O(H'x W'x K^2 x C_{in} x C_{out})$$
- Memory usage is proportional to feature maps.


####  3. 3D CNN (3D Convolutional Neural Network)
3D CNNs extend 2D convolutions by adding a depth dimension, making them suitable for volumetric data (e.g., video, medical scans).

**Advantages:**
- Captures spatial and temporal/depth-based features.
- Suitable for volumetric data like MRI scans, CT scans, and video analysis.

**Disadvantages:**
- Significantly higher computational cost.
- Requires more memory than 2D CNNs.
- Needs large datasets to generalize well.

**Computational Complexity:**
For an input of size $$(D, H, W, C_{in})$$ and a kernel of size $$(K , K , K, C_{in}, C_{out})$$, the output size is $$(D', H', W', C_{out})$$, and the complexity is:
- $$O(D' x H' x W' x K^3 x C_{in} x C_{out})$$
- Memory requirements are significantly higher than 2D CNNs due to additional depth dimension.

### Performance Analysis
After passing through 2D CNN or 3D CNN, the embedding is fed into an MLP (Multi-Layer Perceptron) to fine-tune the figure, ensuring that it fits well with the training set of the dataset. The MLP acts as a feature adjuster, refining the extracted representations to improve model performance on the given data.

Here are the performance figures over epochs for 2D and 3D CNN when running on the MedMNIST dataset.
<img src="assets/2DCNN_trained.png" width="500">

The training and validation loss increase instead of decreasing, indicating that the model struggles to learn meaningful features. After the first few epochs, the loss fluctuates and stabilizes at a high value (~1.8), suggesting that the model is not converging well. The test loss (1.7170) is high, implying poor generalization. This suggests that the 2D CNN may not be capturing the necessary spatial information effectively.


<img src="assets/3DCNN_trained.png" width="500">

The training and validation loss decrease rapidly in the first few epochs, indicating effective learning. The loss stabilizes at a low value (~0.2041), suggesting that the model has converged well. Some fluctuations in validation loss are observed, but overall, it remains relatively low. The low test loss (0.2041) suggests good generalization, making this model more suitable for the dataset.

**Conclusion:**
- The 3D CNN significantly outperforms the 2D CNN in this case, achieving much lower loss values.
- The 2D CNN struggles to learn useful features, leading to poor convergence.
- Since MedMNIST contains medical imaging data, 3D CNNs may be better suited for capturing spatial relationships in volumetric data.


## Machine Learning Algorithm 
### Decision tree
**Performance:** With each model, the depth coefficients in the range from 5 to 30 with steps of 5 will be substituted for the performance analysis. As the result, the graph showing the correlation between the depth coefficient and performance will be plotted with the accuracy at each step.

Flatten: 
 
| Max Depth | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| 5         | 0.783651 | 0.808604  | 0.783651 | 0.782979 |
| 10        | 0.847283 | 0.854890  | 0.847283 | 0.846956 |
| 15        | 0.852178 | 0.857412  | 0.852178 | 0.851908 |
| 20        | 0.862947 | 0.865628  | 0.862947 | 0.862577 |
| 25        | 0.869799 | 0.871304  | 0.869799 | 0.869791 |

2D_CNN_init:

| Max Depth | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| 5         | 0.883505 | 0.883267  | 0.883505 | 0.883099 |
| 10        | 0.900636 | 0.900718  | 0.900636 | 0.900612 |
| 15        | 0.903573 | 0.903768  | 0.903573 | 0.903510 |
| 20        | 0.895742 | 0.895717  | 0.895742 | 0.895608 |
| 25        | 0.894763 | 0.894838  | 0.894763 | 0.894700 |

2D_CNN_pretrained:
 
| Max Depth | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| 5         | 0.861968 | 0.861571  | 0.861968 | 0.860995 |
| 10        | 0.891826 | 0.891916  | 0.891826 | 0.891008 |
| 15        | 0.877141 | 0.876791  | 0.877141 | 0.875809 |
| 20        | 0.879589 | 0.879295  | 0.879589 | 0.878279 |
| 25        | 0.881547 | 0.880992  | 0.881547 | 0.880747 |

3D_CNN_init:
 
| Max Depth | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| 5         | 0.906510 | 0.907809  | 0.906510 | 0.906656 |
| 10        | 0.929515 | 0.929580  | 0.929515 | 0.929477 |
| 15        | 0.926089 | 0.926046  | 0.926089 | 0.925962 |
| 20        | 0.924131 | 0.924283  | 0.924131 | 0.924021 |
| 25        | 0.918747 | 0.918625  | 0.918747 | 0.918564 |

 3D_CNN_pretrained:
 
| Max Depth | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| 5         | 0.995595 | 0.995602  | 0.995595 | 0.995595 |
| 10        | 0.994126 | 0.994189  | 0.994126 | 0.994132 |
| 15        | 0.994126 | 0.994189  | 0.994126 | 0.994132 |
| 20        | 0.994126 | 0.994189  | 0.994126 | 0.994132 |
| 25        | 0.994126 | 0.994189  | 0.994126 | 0.994132 |

<img src="assets/f1_graph.png" width="800">


**Analysis:** 

Performance: The 3D_CNN_pretrained model performs the best, staying close to 1 at all depths, meaning it makes almost perfect predictions. The 3D_CNN_init model also does well but drops slightly at deeper depths. The 2D_CNN_init and 2D_CNN_pretrained models have good F1 scores, but the pretrained version peaks early and then declines a bit, possibly due to overfitting. The flatten model starts with the lowest F1 score but improves with depth, showing that deeper trees help compensate for simpler embeddings. Most models perform best at depths 10-15, with deeper trees not always leading to better results. In general, pretrained models outperform models trained from scratch, and 3D CNN models work better than 2D CNN models, proving that richer feature extraction leads to better classification.

Storage: As the depth increases, storage requirements grow significantly. A deeper tree has more nodes, which increases memory usage for storing the tree structure and learned parameters. This means that deeper trees require substantially more memory, making them less efficient for large datasets, which tends to store more details about training data, increasing the risk of overfitting.

Speed: As the depth increases, speed decreases in both training and inference. Training takes longer because deeper trees require more splits and calculations. Inference also slows down since each prediction must traverse more levels from the root to a leaf node, increasing the number of comparisons. This added computational cost can make deep trees impractical for large datasets or real-time applications.

### Multi-layer Perceptron (MLP)
In this experiment, we will use an MLP classifier to classify the features from the embeddings mentioned above. From left to right and top to bottom: flatten, 2D init, 2D pretrained, 3D init, and 3D pretrained, respectively.

<img src="assets/flatten_mlp.png" width="250"><img src="assets/2d_init_mlp.png" width="250"><img src="assets/2d_pretrained_mlp.png" width="250"><img src="assets/3d_init_mlp.png" width="250"><img src="assets/3d_pretrained_mlp.png" width="250">

<table>
  <tr>
    <th>Embedding</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>0.2637</td>
    <td>0.1393</td>
    <td>0.2637</td>
    <td>0.1598</td>
  </tr>
  <tr>
    <td>2D_CNN_init</td>
    <td>0.8476</td>
    <td>0.7924</td>
    <td>0.8476</td>
    <td>0.8099</td>
  </tr>
  <tr>
    <td>2D_CNN_pretrained</td>
    <td>0.2699</td>
    <td>0.0730</td>
    <td>0.2699</td>
    <td>0.1149</td>
  </tr>
  <tr>
    <td>3D_CNN_init</td>
    <td>0.2184</td>
    <td>0.0477</td>
    <td>0.2184</td>
    <td>0.0783</td>
  </tr>
  <tr>
    <td>3D_CNN_pretrained</td>
    <td>0.9007</td>
    <td>0.9431</td>
    <td>0.9007</td>
    <td>0.9036</td>
  </tr>
</table>

Overall Observations:

+ 3D Pretrained MLP achieves the best test loss and stability, making it the most promising model. 2D Init MLP has reasonable performance but is outperformed by the 3D pretrained version. 2D Pretrained MLP and Flatten MLP suffer from instability, indicating potential issues with initialization or architecture. 3D Init MLP converges fast but does not generalize well.
  
+ Storage: 
- Flatten MLP (21,952 x 1024 + 1024 parameters in the first layer) → Largest storage requirement
- 3D CNN MLP (11,776 x 1024 + 1024 parameters in the first layer) → Moderate storage
- 2D CNN MLP (1,024 x 1024 + 1024 parameters in the first layer ) → Smallest storage

+ Since all models have the same number of MLP layers, their speed remains identical.
### Naive Bayes

<table>
  <tr>
    <th>Embedding</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>0.7714</td>
    <td>0.7823</td>
    <td>0.7714</td>
    <td>0.7621</td>
  </tr>
  <tr>
    <td>2D_CNN_init</td>
    <td>0.8194</td>
    <td>0.8284</td>
    <td>0.8194</td>
    <td>0.8146</td>
  </tr>
  <tr>
    <td>2D_CNN_pretrained</td>
    <td>0.7186</td>
    <td>0.7311</td>
    <td>0.7186</td>
    <td>0.6826</td>
  </tr>
  <tr>
    <td>3D_CNN_init</td>
    <td>0.8507</td>
    <td>0.8573</td>
    <td>0.8507</td>
    <td>0.8492</td>
  </tr>
  <tr>
    <td>3D_CNN_pretrained</td>
    <td>0.9692</td>
    <td>0.9723</td>
    <td>0.9692</td>
    <td>0.9692</td>
  </tr>
</table>

In this section, we discuss about the Naive Bayes algorithm and specifically its Gaussian variant. we utilize the Scikit-learn library with default parameters to fit and make predictions on the embeddings. The table above shows that the 3D embeddings, regardless of whether its before or after pretraining, performs significantly better than the other embedding methods. This is expected as the original data were 3-dimensional. However, an interesting observation shows that 2D_CNN_pretrained has the worst performance, likely due to a dimensional mismatch.

The Gaussian Naive Bayes formula:

$$P(x_{i}\mid y)=\frac{1}{\sqrt{2\pi \sigma_y^{2}}} \exp \left(-\frac{(x_{i} -\mu_{y})^2}{2\sigma_y^{2}} \right)$$

In terms of sparse feature handling, Gaussian Naïve Bayes (GNB) is not ideal for handling sparse features because it assumes a Gaussian distribution, which may not be well-suited for high-dimensional, sparse data.Flattened embeddings likely resulted in a less structured representation, leading to an accuracy of 77.14%. 3D CNN-based embeddings performed significantly better, likely because 3D convolutional networks excel at capturing volumetric information, being able to extract denser, more meaningful features before classification.

Scikit-learn’s GaussianNB includes a var_smoothing parameter (default: 1e-9) to stabilize variance estimates when features have very small variances. For low-variance features, GaussianNB may overfit, leading to unstable predictions. Model trained 2D_CNN_pretrained embeddings (71.86% accuracy) might have suffered due to a mismatch between pretrained features and the dataset, leading to variance underestimation. Smoothing helps prevent division by very small variances, which might explain why performance did not drop further.

Conditional Independence between features given the class is naturally presumed by Gaussian Naive Bayes algorith; however, CNN-extracted embeddings are typically highly correlated due to the way it is computed. Flattened embeddings likely had higher independence, making them better suited for GNB than some CNN-pretrained features. 3D CNN-pretrained embeddings (96.92%) had the best performance, suggesting that the extracted features, while correlated, were still structured in a way that aligned well with GNB’s assumptions. 2D_CNN_pretrained performed worse (71.86%), possibly because the extracted features had strong dependencies that GNB could not model effectively.

**Result analysis:** Naïve Bayes works best when features are independent, but CNN-extracted features are often correlated. Despite that, 3D CNN-based models performed the best, possibly because the feature dependencies were still manageable for GNB.


### Graphical Models (Bayesian Networks, HMM)
Graphical models, including Bayesian Networks and Hidden Markov Models (HMMs), are powerful tools when dealing with data that exhibits complex dependencies and uncertainties. However, they are generally infeasible when dealing with high dimensional data, such as for the 3D medical image classification task, even after the data has been flattened or transformed using 2D and 3D CNNs. Consequently, we decided to apply TSNE onto the embeddings to map them into a lower dimensional, so that applying Graphical Models for classification would be more practical. Despite that the models would most likely not perform well due to the conflicting natures of the models as well as the data, we show our findings and observations below:

## Bayesian Networks
Bayesian Networks are graphical models that represent the probabilistic relationships among a set of variables. They leverage graph theory and probability theory, particularly Bayes' theorem, to model uncertainty. A Bayesian network's structure is a Directed Acyclic Graph (DAG) and nodes in the graph represent variables, whereas directed edges (arrows) represent dependencies between those variables. Each node has an associated conditional probability distribution (CPD). This distribution quantifies the probability of a node's value given the values of its "parent" nodes (nodes with edges pointing to it), and the values are updated under the principle of the Bayes' Theorem, hence the name of the algorithm.
For our task, we utilize the Bayesian Networks implementation from pgmpy, which is usually very powerful when applied for certain applications; despite that, it faces significant challenges when applied to MedMNIST 3D data due to the large number of voxels (3D pixels). The number of possible network structures and conditional probability distributions grows exponentially with the number of variables present within the feature space, rendering Bayesian Networks to be  infeasible for doing classification for our task.

Our initial embeddings were originally of very high dimensional space, so we decided to map them to a lower dimensional space using the TSNE dimensional reduction technique. Despite our attempts, the training process still remained unsuccessful, and just doingtraining and inference on the flattened embeddings alone resulted in a timeout error on Google Colab. However, since there were multiple evidence suggesting that Bayesian Networks was unlikely to be effective on our data, judging from the algorithm's inherent nature, we decided to move on to focusing on other machine learning algorithms instead.


## Hidden Markov Machine
Hidden Markov Models (HMMs) are probabilistic sequence models used to analyze and predict sequential data. They're particularly effective when dealing with data where the underlying process is assumed to have hidden, unobserved states. For our task, we take advantage of the GaussianHMM from hmmlearn libary, which is a specific type of HMM that assumes the observed data follows a Gaussian distribution. Typically, how the GaussianHMM would work is that we first have to define the number of hidden states (which is the number of classes) in our task. Then, the GaussianHMM assumes that the observations emitted from each hidden state follow a Gaussian distribution. Therefore, each hidden state is associated with a mean vector and a covariance matrix. 
Hidden Markov Models (HMMs), while fundamentally designed for sequential data, can be adapted to handling classfication tasks, though they might not yield great results due to the inherent nature. The basic principle is to train an HMM for each class in the classification problem. Then, when a new, unknown sequence appears, we calculate the likelihood of that sequence belonging to each of the trained HMMs. The class corresponding to the HMM with the highest likelihood is then assigned as the predicted class. Below are results of applying the GaussianHMM model on the embeddings.

<table>
  <tr>
    <th>Embedding</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>0.5193</td>
    <td>0.6020</td>
    <td>0.5193</td>
    <td>0.5169</td>
  </tr>
  <tr>
    <td>2D_CNN_init</td>
    <td>0.6701</td>
    <td>0.7935</td>
    <td>0.6701</td>
    <td>0.6535</td>
  </tr>
  <tr>
    <td>2D_CNN_pretrained</td>
    <td>0.4968</td>
    <td>0.6635</td>
    <td>0.4968</td>
    <td>0.4617</td>
  </tr>
  <tr>
    <td>3D_CNN_init</td>
    <td>0.6520</td>
    <td>0.7020</td>
    <td>0.6520</td>
    <td>0.6446</td>
  </tr>
  <tr>
    <td>3D_CNN_pretrained</td>
    <td>0.6696</td>
    <td>0.8521</td>
    <td>0.6696</td>
    <td>0.6737</td>
  </tr>
</table>

As expected, the results obtained from the Hidden Markov Machine are unsatisfactory. The model trained on 3D_pretrained embeddings had the best performance out of all the embeddings, though it still falls short compared to machine learning algorithms like Decision Trees or Naive Bayes. Models trained on other embeddings performed rather averagely, with the 2D_CNN_pretrained variant having done extremely poorly with the majority of the metrics being less than 0.5. As I mentioned, these results are more or less expected as HMMs are fundamentally designed for sequential data, and 3D medical images from MedMNIST are inherently volumetric, not sequential, so this results in a conflicting nature between the purpose of the model and the data.


### Support Vector Machine (Classifier)
In the realm of machine learning, Support Vector Machines (SVMs) stand out as robust and versatile algorithms, particularly effective for classification tasks. Imagine there exist data points belonging to two distinct categories, like "potential clients" and "non-clients." SVMs excel at drawing the optimal boundary, or "hyperplane," that best separates these categories.  Instead of just drawing any line that separates the data, SVMs aim to maximize the "margin" – the distance between the hyperplane and the closest data points from each class. These closest points are called "support vectors". However, in most cases, data isn't easily separable by a straight line, this leads to SVMs employing a clever trick called the "kernel trick", which transforms the data into a higher-dimensional space where a linear hyperplane can effectively separate the classes.

When working with the classification task for 3D medical imaging data, CNNs (both 2D and 3D) are instrumental in extracting meaningful features. These features, often represented as high-dimensional vectors, capture complex spatial and temporal patterns within the image. SVCs are inherently designed to handle high-dimensional data efficiently, thanks to the kernel trick, since the relationships between extracted features and medical conditions are often non-linear. SVCs, through kernels like the Radial Basis Function (RBF), can map these features into a higher-dimensional space where linear separation becomes possible. However, there is typically a lot of overhead cost when dealing with high dimensional data, so we also apply the TSNE dimensional reduction method onto the embeddings to save computational cost. The results are shown as below.

<table>
  <tr>
    <th>Embedding</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>0.8267</td>
    <td>0.9060</td>
    <td>0.8267</td>
    <td>0.8421</td>
  </tr>
  <tr>
    <td>2D_CNN_init</td>
    <td>0.9804</td>
    <td>0.9805</td>
    <td>0.9804</td>
    <td>0.9804</td>
  </tr>
  <tr>
    <td>2D_CNN_pretrained</td>
    <td>0.8771</td>
    <td>0.8775</td>
    <td>0.8771</td>
    <td>0.8758</td>
  </tr>
  <tr>
    <td>3D_CNN_init</td>
    <td>0.9736</td>
    <td>0.9736</td>
    <td>0.9736</td>
    <td>0.9735</td>
  </tr>
  <tr>
    <td>3D_CNN_pretrained</td>
    <td>0.9951</td>
    <td>0.9951</td>
    <td>0.9951</td>
    <td>0.9951</td>
  </tr>
</table>

The results from the Support Vector Machine on the embeddings were outstanding, with model trained on 3D_CNN_pretrained embeddings reaching over 99.5% accuracy. Even with the worst models still perform relative well. These results are to be expected, as to how effective SVMs are, especially when they are used in conjunction with CNNs for feature extraction. CNNs can effectively capture spatial information from medical scans, producing rich feature representations, which then, SVCs can learn the complex decision boundaries required to classify these features, leading to accurate diagnoses or other medical predictions. This fearsome combination can lead to a significantly robust medical image classification system.

## References
<a id="1">[1]</a> 
```
@article{yang2023medmnist,
  title={Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
