# CIFAR-10 Image Classification using LeNet-5-inspired Neural Network

This repository contains a neural network for image classification using the CIFAR-10 dataset. The network architecture is inspired by LeNet-5, and it is trained to classify images into one of ten classes. In this README, we'll provide an in-depth overview of the network architecture, the training process, and the evaluation metrics used.

## Network Architecture

The neural network architecture used in this project is inspired by LeNet-5. It consists of several layers designed to extract features and make predictions. The architecture is defined as follows:

1. **Convolutional Layer (Layer 1):**
   - 6 filters
   - 5x5 kernel
   - ReLU activation function

2. **Max Pooling Layer (Layer 2):**
   - 2x2 kernel

3. **Convolutional Layer (Layer 3):**
   - 16 filters
   - 5x5 kernel
   - ReLU activation function

4. **Max Pooling Layer (Layer 4):**
   - 2x2 kernel

5. **Fully Connected Layer (Layer 5):**
   - 120 neurons
   - ReLU activation function

6. **Dropout Layer (Layer 6):**
   - 20% dropout probability

7. **Fully Connected Layer (Layer 7):**
   - 84 neurons
   - ReLU activation function

8. **Dropout Layer (Layer 8):**
   - 20% dropout probability

9. **Fully Connected Layer (Layer 9):**
   - 10 neurons
   - Softmax activation function

The network is trained using the Cross-Entropy Loss function and the Stochastic Gradient Descent (SGD) optimizer.

## Training

The training process is an essential part of building an effective image classification model. Here's an overview of the training procedure:

- The network was trained for 20 epochs, with each epoch representing a complete pass through the training dataset.

- A batch size of 4 was used, which means that the model was updated after processing each batch of 4 images.

- The CIFAR-10 dataset was split into three subsets: the training set, the validation set, and the test set. Approximately 45,000 images were used for training, 5,000 for validation, and the remaining images for testing.

- The Adam optimizer was employed to minimize the Cross-Entropy Loss function. The validation set was used to monitor training progress and to prevent overfitting by selecting the model with the lowest validation loss.

## Evaluation Metrics

After training, the model's performance was evaluated using several key metrics to assess its accuracy and effectiveness in classifying images into the ten CIFAR-10 classes. The following metrics were calculated:

- **Accuracy:** The ratio of correctly classified images to the total number of test images.

- **Precision:** The ratio of true positives to the sum of true positives and false positives.

- **Recall:** The ratio of true positives to the sum of true positives and false negatives.

- **Specificity:** The ratio of true negatives to the sum of true negatives and false positives.

- **F1 Score:** The F1 score is the harmonic mean of precision and recall, providing a balanced measure of model performance.

These metrics provide insights into how well the model is performing and its ability to correctly classify images across the ten classes in the CIFAR-10 dataset.

## Usage

To use this image classification model for your own projects, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using the appropriate package manager for your Python environment.
3. Run the main script to train the model on your machine.
4. Customize the hyperparameters and model architecture to suit your specific use case.

## Run

The primary code for this project can be found in the main script, `main.py`. It includes loading the CIFAR-10 dataset, defining the model architecture, training the model, and evaluating its performance on the test dataset.

```python
python main.py
```

## License

This project is licensed under the MIT License. You can find the full details in the LICENSE.md file.

## Acknowledgments

We would like to express our gratitude to the creators of the CIFAR-10 dataset and the authors of LeNet-5 for their valuable contributions to the field of image classification.

Feel free to explore the code, run experiments, and adapt the model to your own image classification tasks. If you have any questions or suggestions, please don't hesitate to reach out to us.

Happy classifying!

