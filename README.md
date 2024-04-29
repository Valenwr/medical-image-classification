# Medical Image Classification

## Overview
This project involves the development of a deep learning model for classifying medical images into different disease categories. The model is built using TensorFlow and Keras and utilizes a pre-trained VGG16 convolutional neural network architecture for feature extraction.

## Getting Started
### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Google Colab (optional, for running the notebook)

### Installation
You can install the required packages using pip:
\```
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
\```

## Usage
1. Clone this repository to your local machine.
2. Make sure you have access to the medical image dataset.
3. Update the `dataset_dir` variable in the notebook to point to the directory containing your dataset.
4. Open and run the `medical-image-classification.ipynb` notebook using Jupyter or Google Colab.

## Dataset
The dataset consists of medical images categorized into different disease types. It is assumed that the dataset is stored in a directory structure where each disease type has its own subdirectory containing the respective images.

## Model Architecture
The model architecture involves the following steps:
- Loading the pre-trained VGG16 model without the top classification layer.
- Freezing the convolutional base to prevent its weights from being updated during training.
- Adding custom classification layers on top of the VGG16 base.
- Compiling the model with the Adam optimizer and categorical cross-entropy loss function.

## Training and Evaluation
- The dataset is split into training, validation, and test sets using a specified ratio.
- The model is trained on the training set and evaluated on the validation set.
- Training stops early if the model achieves a specified accuracy threshold.
- Loss and accuracy curves are plotted to visualize model performance.
- The trained model is evaluated on the test set, and additional evaluation metrics are calculated, including accuracy, recall, and ROC AUC score.
- Confusion matrix and heatmap are generated to visualize the model's classification performance.

## Credits
- Dataset Source: [Medical Imaging Original Dataset](https://www.kaggle.com/datasets/heartzhacker/medical-imaging)

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
