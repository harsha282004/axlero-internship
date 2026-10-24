# ResNet50 Training and Fine-Tuning Experiment Report

## 1. Objective

The objective of this experiment was to develop and evaluate a deep learning model that classifies chest X-ray images into two categories: NORMAL and PNEUMONIA.

A pretrained ResNet50 model was used as the baseline. A fine-tuning experiment was subsequently conducted to improve classification performance.

## 2. Dataset

The dataset contains 5,856 chest X-ray images.

The patient-level data split was:

* Training set: 4,100 images
* Validation set: 878 images
* Test set: 878 images

The dataset contains 1,583 NORMAL images and 4,273 PNEUMONIA images.

Patient-level splitting was used to reduce the risk of images from the same patient appearing across different data splits.

## 3. Baseline Model

The baseline model used a pretrained ResNet50 architecture.

The final classification layer was adapted for binary classification. The pretrained feature-extraction layers were frozen during baseline training.

The reported configuration included:

* Input image size: 224 × 224 pixels
* Batch size: 8
* Optimizer: Adam
* Learning rate: 0.0001
* Training duration: 2 epochs
* Model selection: Lowest validation loss

The baseline model achieved 93.51% test accuracy and a macro F1-score of 91.32%.

## 4. Fine-Tuning Experiment

The fine-tuning experiment started from the saved baseline checkpoint.

The ResNet50 layer4 block and final classification layer were made trainable. Earlier feature-extraction layers remained frozen.

The optimizer used separate learning rates for the trainable components:

* Layer4 learning rate: 0.00001
* Classification layer learning rate: 0.00005
* Optimizer: Adam
* Batch size: 8
* Training duration: 2 epochs
* Model selection: Lowest validation loss

The fine-tuned model achieved 96.81% test accuracy and a macro F1-score of 95.94%.

## 5. Performance Comparison

| Metric             | Baseline | Fine-tuned |
| ------------------ | -------: | ---------: |
| Accuracy           |   93.51% |     96.81% |
| Macro Precision    |   94.03% |     96.07% |
| Macro Recall       |   89.30% |     95.82% |
| Macro F1-score     |   91.32% |     95.94% |
| Weighted Precision |   93.59% |     96.80% |
| Weighted Recall    |   93.51% |     96.81% |
| Weighted F1-score  |   93.32% |     96.81% |

The fine-tuned model outperformed the baseline across all reported metrics. Macro recall improved by approximately 6.52 percentage points, suggesting a more balanced ability to identify examples from both classes.

## 6. Fine-Tuned Confusion Matrix

The confusion matrix, with rows representing actual classes and columns representing predicted classes, was:

| Actual Class | Predicted NORMAL | Predicted PNEUMONIA |
| ------------ | ---------------: | ------------------: |
| NORMAL       |              222 |                  15 |
| PNEUMONIA    |               13 |                 628 |

The model correctly classified 222 NORMAL images and 628 PNEUMONIA images.

It misclassified 15 NORMAL images as PNEUMONIA and 13 PNEUMONIA images as NORMAL.

These errors demonstrate why evaluation should consider class-specific precision and recall in addition to overall accuracy.

## 7. Training and Validation Curves

The baseline accuracy curves showed improvement over the two training epochs.

During fine-tuning, both training and validation loss decreased:

* Training loss: approximately 0.267 to 0.194
* Validation loss: approximately 0.136 to 0.098

The decreasing validation loss is encouraging, although two epochs are insufficient to establish long-term generalization.

## 8. Limitations

1. The experiment used a limited number of training epochs.
2. The dataset is imbalanced, with more PNEUMONIA images than NORMAL images.
3. The dataset may contain demographic, acquisition, or image-processing biases.
4. High test accuracy does not guarantee generalization to other hospitals or patient populations.
5. The model has not been clinically validated and must not be used to make medical decisions.

## 9. Conclusion

Fine-tuning the pretrained ResNet50 model improved test accuracy from 93.51% to 96.81% and macro F1-score from 91.32% to 95.94%.

The experiment demonstrates the potential benefit of selectively fine-tuning pretrained convolutional neural networks for binary chest X-ray classification.

Further work should investigate robustness across independent datasets, class-specific error patterns, model calibration, and explainability. Any future experiments should use validation data for model selection and preserve the test set for final evaluation.
