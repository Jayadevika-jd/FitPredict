🔥 FitPredict – Machine Learning-Based Calorie Expenditure Prediction

<img width="1912" height="843" alt="Screenshot 2026-11-18 101519" src="https://github.com/user-attachments/assets/5514506b-c903-4c67-8952-9c229bf1eb5f" />
<img width="1572" height="782" alt="Screenshot 2026-11-18 103223" src="https://github.com/user-attachments/assets/70b241f4-e430-40b5-a880-1db9e6f82e38" />
<img width="1447" height="586" alt="Screenshot 2026-11-18 103238" src="https://github.com/user-attachments/assets/690ea11a-c973-45df-94e2-563da9ad7249" />


📌 Project Overview

FitPredict is a machine learning-based calorie expenditure prediction system designed to estimate the number of calories burned during physical exercise.

The project uses personal and physiological information, including age, gender, height, weight, exercise duration, heart rate, and body temperature, to predict calorie expenditure.

A regression-based machine learning approach is used to train and evaluate multiple algorithms. The selected model is integrated into an interactive Streamlit application, allowing users to enter their exercise details and obtain an estimated calorie expenditure.

This project demonstrates an end-to-end machine learning workflow, from data preparation and exploratory analysis to model development, evaluation, serialization, and deployment.

 🎯 Project Objectives

- Understand and analyze exercise-related data.
- Perform data cleaning, validation, and preprocessing.
- Explore relationships between physiological features and calorie expenditure.
- Build and compare multiple regression models.
- Optimize model performance through hyperparameter tuning.
- Evaluate model performance using appropriate regression metrics.
- Develop an interactive prediction application using Streamlit.
- Deploy the application for public access.

📊 Dataset

**Dataset:** [Calorie Burnt 15K – Hugging Face](https://huggingface.co/datasets/mnemoraorg/calorie-burnt-15k)

The dataset consists of two CSV files:

- `raw_exercise.csv`
- `raw_calories.csv`

These datasets were merged using the common `User_ID` column.

**Dataset dimensions after merging:**

- 15,000 observations
- 9 columns

 Input Features

| Feature    | Description                  |
| ---------- | ---------------------------- |
| Gender     | Gender category              |
| Age        | Age of the individual        |
| Height     | Height in centimetres        |
| Weight     | Weight in kilograms          |
| Duration   | Exercise duration in minutes |
| Heart_Rate | Heart rate during exercise   |
| Body_Temp  | Body temperature             |

**Target Variable:** `Calories`

`User_ID` was used only to merge the two datasets and was excluded from model training.

🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the structure, distributions, relationships, and potential anomalies in the dataset.

The analysis included:

- Feature distributions using histograms.
- Outlier exploration using boxplots.
- Relationship analysis using scatterplots.
- Pairwise feature relationships using pairplots.
- Gender-based comparisons.
- Analysis of relationships between exercise duration, heart rate, body temperature, and calorie expenditure.

The EDA findings helped guide preprocessing and model development decisions.

⚙️ Data Preprocessing

The following preprocessing activities were performed:

1. Merged the exercise and calorie datasets using `User_ID`.
2. Checked for missing values and duplicate records.
3. Examined feature distributions and potential outliers.
4. Separated independent variables and the target variable.
5. Applied One-Hot Encoding to the categorical Gender feature.
6. Excluded `User_ID` from the model input features.
7. Split the dataset into training and testing sets using an 80:20 ratio.

The fitted encoder and feature order were saved to ensure consistent preprocessing during application-based predictions.

🤖 Machine Learning Models

The following regression algorithms were implemented and evaluated:

| Model                   | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| Linear Regression       | Baseline regression model                       |
| Ridge Regression        | Linear regression with L2 regularization        |
| Lasso Regression        | Linear regression with L1 regularization        |
| Decision Tree Regressor | Non-linear, tree-based regression               |
| Random Forest Regressor | Ensemble learning using multiple decision trees |

 Hyperparameter Tuning

Hyperparameter optimization was performed for the Decision Tree and Random Forest models.

- GridSearchCV was used for Decision Tree optimization.
- RandomizedSearchCV was used for Random Forest optimization.
- Five-fold cross-validation was used during hyperparameter search.
- Negative Root Mean Squared Error was used as the optimization scoring metric.

The final tuned Random Forest model was selected based on its evaluation performance.

 📈 Model Evaluation and Results

The models were evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted calorie values.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Represents prediction error in the target variable's original unit.
- **R² Score:** Measures the proportion of target variance explained by the model.

Model Comparison

The following results were obtained on the held-out test dataset:

| Model                   |   Test MAE |  Test RMSE |    Test R² |
| ----------------------- | ---------: | ---------: | ---------: |
| Linear Regression       |     8.4415 |    11.4894 |    0.96729 |
| Decision Tree (Default) |     3.4430 |     5.3368 |     0.9929 |
| Decision Tree (Tuned)   |     3.3459 |     5.0751 |     0.9936 |
| Random Forest (Default) |     1.7139 |     2.6788 |    0.99822 |
| Random Forest (Tuned)   | **1.6910** | **2.6511** | **0.9983** |

Ridge and Lasso regression produced results very close to Linear Regression.

Selected Model: Tuned Random Forest Regressor

The tuned Random Forest model achieved:

| Evaluation Metric     |  Result |
| --------------------- | ------: |
| Test MAE              |  1.6910 |
| Test MSE              |  7.0287 |
| Test RMSE             |  2.6511 |
| Test R²               |  0.9983 |
| Cross-validation RMSE | 3.06794 |

The tuned Random Forest model demonstrated the strongest overall test performance among the evaluated models.

**Note:** These results represent performance on the selected dataset and test split. They should not be interpreted as proof of equivalent accuracy on real-world exercise measurements.

 🌳 Feature Importance

Feature importance was examined using the tree-based models to understand which input variables contributed most to their predictions.

Feature importance provides insight into the model's predictive patterns. It does not establish a causal relationship between a feature and calorie expenditure.
🔬 Residual and Error Analysis

Residual analysis was performed to examine prediction errors and identify potential patterns.

The analysis included:

- Residual distribution.
- Residuals versus predicted values.
- Prediction error across different calorie ranges.

The analysis showed that absolute prediction errors tended to increase across higher calorie ranges in the test dataset. This is an important consideration when interpreting model performance.

 💾 Model Serialization

The trained model and preprocessing components were saved using Joblib.

The following artifacts were stored:

- `random_forest_model.pkl` – Trained Random Forest model.
- `gender_encoder.pkl` – Fitted One-Hot Encoder.
- `model_features.pkl` – Expected model input feature order.

Git LFS was used to manage large model artifacts in the GitHub repository.

 🌐 Streamlit Web Application

The trained model was integrated into an interactive application developed using Streamlit.

Application Workflow

1. User enters exercise-related information.
2. Gender is transformed using the saved encoder.
3. Input features are arranged in the expected model feature order.
4. The trained Random Forest model generates a prediction.
5. The estimated calorie expenditure is displayed.

**Live Application:**https://fitpredict-mcbyyunzcyn7rdvfuggsaw.streamlit.app/

 Application Features

- Interactive input fields.
- Gender selection.
- Exercise and physiological feature inputs.
- Real-time prediction on submission.
- Estimated calorie expenditure displayed in kcal.

🛠️ Technologies and Tools

| Technology       | Usage                                 |
| ---------------- | ------------------------------------- |
| Python           | Programming language                  |
| Pandas           | Data manipulation                     |
| NumPy            | Numerical operations                  |
| Matplotlib       | Data visualization                    |
| Seaborn          | Exploratory data visualization        |
| Scikit-learn     | Machine learning and model evaluation |
| Joblib           | Model serialization                   |
| Streamlit        | Interactive web application           |
| Jupyter Notebook | Data analysis and model development   |
| VS Code          | Application development               |
| Git and GitHub   | Version control and project hosting   |
| Git LFS          | Large model artifact storage          |

 📁 Project Structure

```text
FitPredict/
│
├── Notebooks/
│   └── Fitpredict_preprocessing.ipynb
│
├── Processed/
│   └── fitpredict_validated.csv
│
├── Models/
│   ├── random_forest_model.pkl
│   ├── gender_encoder.pkl
│   └── model_features.pkl
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

 🚀 Run the Project Locally

### 1. Clone the repository


git clone https://github.com/Jayadevika-jd/FitPredict.git


 2. Navigate to the project directory


cd FitPredict


### 3. Install dependencies


pip install -r requirements.txt


### 4. Run the Streamlit application


streamlit run app.py


The application will open in your browser.

**Note:** Git LFS must be installed and the model artifacts must be downloaded successfully when cloning the repository.

 ⚠️ Limitations and Disclaimer

- The application provides model-based estimates rather than directly measured calorie expenditure.
- Prediction quality depends on the characteristics and limitations of the training dataset.
- Dataset patterns may not fully represent real-world exercise conditions or individual physiological differences.
- High test-set performance does not guarantee equivalent performance on unseen real-world populations.
- The application is an educational machine learning project and is not a medical device.
- Predictions should not be used for medical decisions or as a substitute for professional fitness or healthcare advice.

🔮 Future Improvements

- Validate model performance using independent real-world exercise data.
- Explore additional physiological and activity-related features.
- Investigate model generalization and potential data leakage.
- Improve input validation and user experience.
- Add prediction explanations and model interpretability.
- Compare additional ensemble and boosting algorithms.
- Expand model monitoring and performance evaluation.

👩‍💻 Author

Jayadevika.G.L

Machine Learning Project – FitPredict

GitHub: [Jayadevika-jd](https://github.com/Jayadevika-jd)

