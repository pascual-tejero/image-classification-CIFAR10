# CIFAR-10 Image Classification using LeNet-5-inspired Neural Network and ResNet-18

This project is an image classification task using the CIFAR-10 dataset. We provide implementations of two different neural network architectures, LeNet-5 and ResNet-18, for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Project Structure

The project directory is organized as follows:

- **commands/**

  This folder contains two shell script files for managing different aspects of the project.

  - `train_model.sh`: Use this script to initiate the training of a neural network. It provides options for selecting the neural network architecture and adjusting hyperparameters such as learning rate, batch size, and the number of epochs. For example:

    ```bash
    python main.py --model_name "LeNet5" --epochs 20 --batch_size 4 --lr 0.001
    python main.py --model_name "ResNet18" --epochs 20 --batch_size 4 --lr 0.001
    ```

  - `visualize_data.sh`: This script is designed for visualizing images in the dataset. You can run the script to display images and save them if desired. It works in the following way:

    ```bash
    python visualize_data.py --visualize --save
    ```

- **models/**

  The `models` folder contains two Python files:

  - `LeNet5.py`: This file holds the implementation of a LeNet-5 inspired neural network. You can explore the contents of this file to learn more about its architecture.

  - `ResNet18.py`: In this file, you'll find the implementation of a ResNet-18 model. Details about its architecture and components can be found inside this file.

- **results/**

  The `results` folder is used to store result files in .txt format. These files include metrics such as true positives (tp), false positives (fp), true negatives (tn), false negatives (fn), recall, precision, specificity, and f1-score. These metrics help assess the performance of the trained models on the test set.

- **saved_models/**

  The `saved_models` directory contains the saved model weights in .pth file format. Each model file represents the best-performing model, selected based on the lowest validation cross-entropy score during training.

- **utils/**

  This folder houses a Python script named `visualize_data.py`. This script allows you to visualize and optionally save images from the dataset. You can explore the script to understand its functionality in more detail.

- **main.py**

  The root directory contains the `main.py` script, which serves as the core of the project. It's responsible for loading the CIFAR-10 dataset, configuring hyperparameters, initiating the training and validation process, and performing the testing stage for the neural networks.

This well-structured project directory makes it easy to manage and experiment with different neural network architectures, hyperparameters, and data visualization while working on your CIFAR-10 image classification task.

## Network Architecture

### LeNet-5

The LeNet-5 architecture is designed for image classification tasks. It consists of the following layers:

1. Convolutional layer (6 filters, 5x5 kernel, ReLU activation function)
2. Max pooling layer (2x2 kernel)
3. Convolutional layer (16 filters, 5x5 kernel, ReLU activation function)
4. Max pooling layer (2x2 kernel)
5. Fully connected layer (120 neurons, ReLU activation function)
6. Dropout layer (20% dropout probability)
7. Fully connected layer (84 neurons, ReLU activation function)
8. Dropout layer (20% dropout probability)
9. Fully connected layer (10 neurons, Softmax activation function)

The LeNet-5 network uses the Cross-Entropy Loss function and the Adam optimizer.

### ResNet-18

ResNet-18 is a deeper convolutional neural network that has shown excellent performance in image classification. It consists of the following layers:

1. Initial Convolutional layer (64 filters, 7x7 kernel, stride=2, padding=3)
2. Batch normalization
3. 4 Residual Blocks, each containing two convolutional layers with batch normalization and ReLU activation
4. Global Average Pooling layer
5. Fully connected layer (10 neurons, Softmax activation function)

The ResNet-18 network also uses the Cross-Entropy Loss function and the Adam optimizer.

## Training

The training process is an essential part of building an effective image classification model. Here's an overview of the training procedure:

- The network was trained for 20 epochs, with each epoch representing a complete pass through the training dataset.

- A batch size of 4 was used, which means that the model was updated after processing each batch of 4 images.

- The CIFAR-10 dataset was split into three subsets: the training set, the validation set, and the test set. Approximately 45,000 images were used for training, 5,000 for validation, and the remaining images for testing.

- The Adam optimizer was employed to minimize the Cross-Entropy Loss function. The validation set was used to monitor training progress and to prevent overfitting by selecting the model with the lowest validation loss.

## Results

After training, the model's performance was evaluated using several key metrics to assess its accuracy and effectiveness in classifying images into the ten CIFAR-10 classes. The following metrics were calculated:

- **Accuracy:** The ratio of correctly classified images to the total number of test images.

- **Precision:** The ratio of true positives to the sum of true positives and false positives.

- **Recall:** The ratio of true positives to the sum of true positives and false negatives.

- **Specificity:** The ratio of true negatives to the sum of true negatives and false positives.

- **F1 Score:** The F1 score is the harmonic mean of precision and recall, providing a balanced measure of model performance.

These metrics provide insights into how well the model is performing and its ability to correctly classify images across the ten classes in the CIFAR-10 dataset.

Here are the results obtained for each network:

|                   | LeNet5     | ResNet18   |
|-------------------|------------|------------|
| True Positives (TP) | 441        | 618        |
| True Negatives (TN) | 387        | 473        |
| False Positives (FP) | 305        | 430        |
| False Negatives (FN) | 418        | 208        |
| **Accuracy**          | 0.534      | 0.631      |
| **Precision**         | 0.591      | 0.590      |
| **Recall**            | 0.513      | 0.748      |
| **Specificity**       | 0.559      | 0.524      |
| **F1 Score**          | 0.550      | 0.660      |


## Usage

To use this image classification model for your own projects, follow these steps:

1. Clone this repository to your local machine.
2. Create and activate a conda environment:
   ```bash
   conda create --name im_class_CIFAR10
   conda activate myenv
   ```
4. Install the required dependencies by using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```
5. Navigate to the `commands` directory to manage different aspects of the project.

   - To initiate the training of a neural network, use `train_model.sh`. It provides options for selecting the neural network architecture and adjusting hyperparameters. For example:
     ```bash
     cd commands
     bash train_model.sh
     ```
   - Use `visualize_data.sh` in the `commands` directory for visualizing images in the dataset. You can run the script to display images and save them if desired. It works as follows:
     ```bash
     cd commands
     bash visualize_data.sh
     ```

6. Explore the `models` directory, which contains two subfolders: `LeNet5/` and `ResNet18/`. Each folder holds the implementation details of its respective neural network architecture.

7. Check the `results` directory for result files in .txt format, including metrics such as true positives, false positives, true negatives, false negatives, recall, precision, specificity, and F1-score. These metrics help assess the performance of the trained models on the test set.

8. The best-performing model weights can be found in the `saved_models` directory in .pth file format.

9. If you want to visualize images from the dataset, explore the `utils` folder. It contains a Python script named `visualize_data.py` that allows you to visualize and optionally save images.


## License

This project is licensed under the MIT License. You can find the full details in the LICENSE.md file.

## Acknowledgments

I extend my gratitude to the creators of the CIFAR-10 dataset, whose efforts have made valuable contributions to the field of image classification. Additionally, I appreciate the advancements and contributions of various researchers and institutions that have influenced the development of image classification techniques.

Feel free to explore the code, run experiments, and adapt the model to your specific image classification tasks.

Happy classifying!

