# Brain Tumor Classification using CNN (97% Accuracy)

## Project Overview and Purpose
This project implements a Deep Learning solution to classify brain tumors from MRI scans into four categories: Glioma, Meningioma, No Tumor, and Pituitary. Early and accurate detection is critical in oncology, and this project explores how Transfer Learning can be used to achieve clinical-grade accuracy.

## Key Technologies and Libraries
- **Frameworks**: TensorFlow, Keras
- **Computer Vision**: OpenCV (`cv2`), ImageDataGenerator
- **Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## Model Architecture and Methodology
This project utilizes **Transfer Learning** by evaluating several pre-trained architectures:
- **ResNet50V2**
- **VGG16 & VGG19**
- **MobileNetV2**
- **EfficientNetB3**

### Workflow:
1. **Preprocessing**: Images were resized to (224, 224), normalized, and augmented to prevent overfitting.
2. **Training**: Used the Adam optimizer with a categorical cross-entropy loss function.
3. **Callbacks**: Implemented `EarlyStopping` and `ReduceLROnPlateau` to optimize training time and prevent divergence.

## Results and Insights
- **Top Performer**: The best-performing model achieved an accuracy of **~97%** on the validation set.
- **Visualizations**: The project includes Confusion Matrices and Classification Reports to visualize precision/recall for each tumor type.
- **Comparison**: A horizontal bar chart compares the final accuracy across all tested architectures.



## How to Run Locally
1. **Clone the repo**: 
   ```bash
   git clone <your-repo-url>
