# Tourism Experience Analytics

## Classification, Rating Prediction and Recommendation System

A machine learning project for analyzing tourism experiences using transaction, user and attraction data.

## Project Objectives

1. **Regression:** Predict the rating a user may give to a tourist attraction.
2. **Classification:** Predict the user's visit mode such as Business, Family, Couples or Friends.
3. **Recommendation:** Recommend tourist attractions using historical ratings and popularity.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Google Colab

## Project Structure

```text
Tourism-Experience-Analytics/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── Tourism_Experience_Analytics_Google_Colab.ipynb
└── assets/
    └── screenshots/
```

## Dataset

The project uses tourism-related datasets such as:

- Transaction data
- User data
- City data
- Attraction/item data
- Attraction type data
- Visit mode data
- Continent data
- Country data
- Region data

Run the Google Colab notebook to clean, merge and preprocess the datasets.

## Installation

Clone or download this repository.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run:

```bash
streamlit run app.py
```

The application provides:

- Tourism dataset analytics
- Attraction rating prediction
- Visit mode prediction
- Personalized attraction recommendations

## Model Files

After running the notebook, place the generated files inside the `models/` folder:

```text
models/
├── best_regression_model.joblib
└── best_classification_model.joblib
```

Place the cleaned dataset here:

```text
data/
└── cleaned_tourism_dataset.csv
```

## Google Colab Workflow

1. Open the notebook in the `notebooks/` folder.
2. Upload all tourism dataset files.
3. Run the notebook cells from top to bottom.
4. Download the cleaned dataset and trained model files.
5. Add them to the GitHub project folders.
6. Run the Streamlit application.

## Evaluation Metrics

### Regression
- MAE
- RMSE
- R² Score

### Classification
- Accuracy
- Precision
- Recall
- F1 Score

## Future Improvements

- Collaborative filtering
- Content-based recommendation
- Hybrid recommendation system
- Advanced visualizations
- Cloud deployment

## Author

Vaishnavi

## License

This project is created for educational and academic purposes.
