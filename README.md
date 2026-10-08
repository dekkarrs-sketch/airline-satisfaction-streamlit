# Predicting Airline Passenger Satisfaction

## A Machine-Learning Application Using Streamlit

This student project demonstrates a complete machine-learning workflow using Python, scikit-learn, and Streamlit. The application loads the **Airline Passenger Satisfaction dataset (about 130,000 survey records)**, cleans it, trains three classification models (Logistic Regression, Decision Tree, Random Forest), compares their performance, checks for overfitting, identifies the main drivers of satisfaction, and provides an interactive prediction demonstration.

> **Important:** This project uses a public Kaggle survey dataset. Predictions describe patterns in that dataset only. They are not guarantees about any real airline or passenger and must not be used to make decisions about individuals.



## Project Objective

The project investigates the following research question:

> Which passenger, travel, delay, and in-flight service variables best predict whether a passenger is satisfied with their flight?

The purpose is to show how to:

- Load, combine, and clean a real-world dataset
- Handle missing values and duplicate rows
- Prepare numeric and categorical variables
- Divide data into training and testing sets
- Train and compare several classification models
- Evaluate models with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Check for overfitting
- Interpret feature importance for business recommendations
- Build an interactive Streamlit interface

## Project Structure

```
airline-satisfaction-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── Airline_Satisfaction_Prediction.ipynb
└── data/
    ├── train.csv
    └── test.csv
```

`app.py` contains the data loading, preprocessing, model training, evaluation, and Streamlit interface. The `data` folder holds the Kaggle files, and the notebook is the original analysis (including k-fold cross-validation).

## Dataset

Source: [Airline Passenger Satisfaction (Kaggle)](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction), originally from a US airline customer survey. `train.csv` and `test.csv` are combined and re-split by the app (about 129,880 rows after removing duplicates). The variables include:

| Variable | Description |
| --- | --- |
| `Gender` | Passenger gender |
| `Customer Type` | Loyal or disloyal customer |
| `Age` | Passenger age |
| `Type of Travel` | Business or personal travel |
| `Class` | Business, Eco Plus, or Eco |
| `Flight Distance` | Distance of the flight |
| `Departure Delay in Minutes` | Departure delay |
| `Arrival Delay in Minutes` | Arrival delay |
| 14 service ratings | Wifi, time convenience, online booking, gate location, food and drink, online boarding, seat comfort, entertainment, on-board service, leg room, baggage handling, check-in, inflight service, cleanliness (0-5) |
| `satisfaction` | Target: satisfied / neutral or dissatisfied |

The target has two classes:

- `Satisfied`
- `Neutral or dissatisfied`

## Machine-Learning Method

Three models are compared:

- **Logistic Regression** - interpretable baseline
- **Decision Tree** (max depth 8) - captures non-linear rules, easy to explain
- **Random Forest** (200 trees, max depth 12) - ensemble model, usually the strongest

The workflow is:

1. Load and combine `train.csv` and `test.csv`; drop the `id` column.
2. Replace missing `Arrival Delay in Minutes` values with the median.
3. Remove duplicate rows.
4. Divide the dataset into 80% training and 20% testing data (random seed 42, stratified).
5. Standardize numeric variables.
6. One-hot encode categorical variables.
7. Train the three classifiers.
8. Evaluate the models using unseen testing data.
9. Run overfitting checks and rank feature importance.

*Note:* the original notebook label-encodes categories; the app uses one-hot encoding inside a scikit-learn pipeline, which is better suited to logistic regression. Results are very similar.

## Evaluation Measures

The application presents:

- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrices and ROC curves for all three models
- Classification report
- Train vs test accuracy gap
- Decision-tree depth sensitivity
- Random Forest feature importance

Using the fixed split, the results are approximately:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 87.4% | 0.870 | 0.835 | 0.852 | 0.927 |
| Decision Tree | 93.5% | 0.935 | 0.914 | 0.924 | 0.984 |
| **Random Forest** | **95.3%** | 0.951 | 0.941 | **0.946** | **0.992** |

**Overfitting check:** the train-test accuracy gaps are small (Logistic Regression 0.04 points, Decision Tree 0.3 points, Random Forest 0.8 points). An unrestricted decision tree reaches 100% training accuracy but only 94.6% test accuracy, which shows why a depth limit is used.

**Top drivers of satisfaction (Random Forest):** Online boarding, Inflight wifi service, Business class, Type of travel (personal vs business), and Inflight entertainment.

## Streamlit Pages

The application contains four tabs:

1. **Overview** - explains the project, models, and responsible-use conditions.
2. **Explore data** - satisfaction distribution, satisfaction rate by category, selectable numeric distributions, correlation chart, and sample records.
3. **Prediction demo** - enter one passenger profile and choose a model to see the prediction and probability.
4. **Model results** - model comparison, confusion matrices, ROC curves, classification report, overfitting checks, and feature importance.

## Installation and Local Use

### 1. Download or clone the repository

```
git clone https://github.com/YOUR-USERNAME/airline-satisfaction-streamlit.git
cd airline-satisfaction-streamlit
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```
pip install -r requirements.txt
```

### 4. Run the application

```
streamlit run app.py
```

The application should open at `http://localhost:8501`. The first start takes about 30-60 seconds while the three models train; later interactions are fast.

## requirements.txt

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data` folder (with `train.csv` and `test.csv`). If the browser upload is slow, use Git instead.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select **Create app**.
5. Select the GitHub repository and branch.
6. Enter `app.py` as the main file path.
7. Select **Deploy**.
8. Open the deployed application and test every tab.

## Business Recommendations

- Prioritize improving **online boarding** and **inflight wifi**, the two strongest drivers of satisfaction.
- Investigate why **personal-travel** and **Eco-class** passengers report lower satisfaction and tailor service for them.
- Improve inflight entertainment and seat comfort, which also rank highly.
- Treat delays as a secondary factor in this dataset, but keep monitoring them.

## Limitations

- Labels come from survey responses, which reflect opinions at one moment in time.
- Data comes from one airline context and may not generalize to other airlines or regions.
- The data has no seasonality, route, or price information.
- Feature importance shows association, not causation.
- Very high accuracy partly reflects strong, direct service-rating variables.
- The models have not been validated on new, independently collected data.

## Future Improvements

- Tune hyperparameters with cross-validation (GridSearchCV)
- Try gradient boosting (XGBoost, LightGBM)
- Add model-explanation tools such as SHAP
- Collect newer data and test on a different airline
- Add fairness checks across gender and age groups

## Suggested Screenshots for Submission

1. Application overview
2. Data-exploration chart
3. Passenger prediction
4. Model comparison table and chart
5. Confusion matrices
6. ROC curves
7. Overfitting checks
8. Feature importance chart

## Suggested Questions to Answer

1. Why were three different models compared?
2. Why is the dataset divided into training and testing sets?
3. What does one-hot encoding do?
4. Why are numeric variables standardized, and why is the median used for missing delays?
5. What is the difference between accuracy and recall?
6. What does ROC-AUC measure?
7. How do we know the high accuracy is not overfitting?
8. Which features drive satisfaction, and what should the airline do about them?
9. What are the limitations of survey-based data?
10. How could the model be improved in future research?


