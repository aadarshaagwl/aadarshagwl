# Helmet Detection using Deep Learning

## Project Overview

This project focuses on using deep learning techniques for real-time helmet detection in images. It aims to classify whether a person is wearing a helmet or not. The project uses a Convolutional Neural Network (CNN) to train a model on a dataset containing labeled images of people with and without helmets. The trained model can be used for live predictions using a webcam feed, making it applicable in real-time scenarios such as traffic monitoring and safety systems.

### Key Features:
- **Helmet Detection:** The model can identify whether a person is wearing a helmet or not from images.
- **Real-Time Processing:** The model has been designed to make predictions on live webcam data.
- **Model Architecture:** A CNN-based architecture is used to classify images into "With Helmet," "Without Helmet," and "Not Sure."

## Dataset

The dataset used for this project consists of images categorized into three classes:
- **With Helmet**
- **Without Helmet**
- **Not Sure**

You can access the dataset from the following link:

[Helmet Detection Dataset on Kaggle](https://www.kaggle.com/datasets/aadarshagwl/helmet-detection-dataset)

The dataset was preprocessed by cropping images based on bounding box annotations, followed by augmentation techniques to increase the robustness of the model.

## How to Use

1. Clone this repository.
2. Install the necessary dependencies from the `requirements.txt` file.
3. Load the trained model (`demo_new_helmet_detection_cnn.h5`).
4. Run the code to start live webcam helmet detection or use any static image for prediction.

## Kaggle Notebook

You can access the Kaggle notebook used to train the model, including the dataset and code, from the following link:

[Kaggle Notebook - Helmet Detection](https://www.kaggle.com/code/aadarshagwl/demo-version-50epoch/edit/run/217611098)

## Results

The model achieved **83.33% accuracy** on the test set. The following metrics were recorded during training:

- Training accuracy reached as high as 85%.
- Validation accuracy and loss were closely monitored to avoid overfitting, and early stopping was applied to prevent unnecessary training.

## Conclusion

The helmet detection model can be further improved by handling multiple object detection, optimizing the model architecture, and expanding the dataset for a wider range of scenarios. This project demonstrates the potential of deep learning for real-time safety applications.

## Future Work

1. **Real-Time Multiple Object Detection:** Enhance the model to detect helmets in multiple people within the same frame.
2. **Handling Class Imbalance:** Implement techniques to handle class imbalance for more accurate results.
3. **Integration with Traffic Monitoring Systems:** The model can be integrated into real-time traffic safety systems for automated helmet detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
