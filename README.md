# Heart Disease Prediction System

This project implements a machine learning system to predict the likelihood of heart disease based on various health parameters. The system uses multiple machine learning models (Logistic Regression, Random Forest, and K-Nearest Neighbors) and provides both a command-line interface and a web interface using Streamlit.

## Features

- Data preprocessing and exploration
- Multiple machine learning models for comparison
- Model evaluation metrics (accuracy, precision, recall, F1-score)
- Interactive web interface using Streamlit
- Feature importance visualization for Random Forest model
- Support for custom user input

## Requirements

The project requires Python 3.7+ and the following packages:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.10.0
```

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

To run the basic model training and evaluation:

```bash
python cardiac_disease_prediction.py
```

This will:
- Load the dataset
- Perform exploratory data analysis
- Train multiple models
- Show evaluation metrics
- Save visualization plots

### Web Interface

To run the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a web browser where you can:
- Input patient health parameters
- Get instant predictions
- View prediction probabilities
- See feature importance (for Random Forest model)

## Input Features

The model uses the following features for prediction:

1. Age (years)
2. Sex (Male/Female)
3. Chest Pain Type (4 categories)
4. Resting Blood Pressure (mm Hg)
5. Cholesterol (mg/dl)
6. Fasting Blood Sugar > 120 mg/dl (Yes/No)
7. Resting ECG Results (3 categories)
8. Maximum Heart Rate
9. Exercise Induced Angina (Yes/No)
10. ST Depression
11. Slope of Peak Exercise ST Segment
12. Number of Major Vessels
13. Thalassemia (3 categories)

## Model Performance

The system trains and compares multiple models:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors

The best performing model is automatically selected for making predictions in both interfaces.

## Visualizations

The system generates two visualization files:
- `correlation_matrix.png`: Shows the correlation between different features
- `target_distribution.png`: Shows the distribution of heart disease cases in the dataset

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
