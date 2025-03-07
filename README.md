
# Health Insurance Cost Prediction

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Project Structure](#project-structure)  
4. [Dataset Information](#dataset-information)  
5. [Technologies & Libraries Used](#technologies--libraries-used)  
6. [Setup & Installation](#setup--installation)  
7. [Usage](#usage)  
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
9. [Model Building & Evaluation](#model-building--evaluation)  
10. [Deployment](#deployment)  
11. [Results & Insights](#results--insights)  
12. [Future Scope](#future-scope)  
13. [References](#references)  
14. [License](#license)  

---

## 1. Project Overview
This repository contains a **Health Insurance Cost Prediction** project. The goal is to **predict individual health insurance premiums (charges)** based on various factors such as **age, sex, BMI, number of children, smoking status, and region**. By applying **machine learning** and **data analysis** techniques, we aim to provide an interactive and accurate way for insurance companies and individuals to estimate healthcare costs.

### Objectives
- **Accurately predict** insurance charges based on user inputs.  
- **Identify key factors** that drive insurance costs (e.g., smoking status, age, BMI).  
- Provide a **user-friendly web interface** for real-time predictions using Flask.  

This project includes:
- **Data exploration and visualization** to understand the underlying trends.  
- **Machine learning models** (Linear Regression, Ridge, SVR, Random Forest) to find the best predictive algorithm.  
- A **Flask web application** that allows users to enter their details and obtain a predicted insurance cost.

---

## 2. Key Features
1. **Interactive Web Application**  
   - Simple user interface built with Flask, HTML, CSS, and JavaScript.  
   - Users can input their information and immediately get predicted charges.

2. **Multiple ML Models Compared**  
   - Linear Regression  
   - Support Vector Regression (SVR)  
   - Ridge Regression  
   - Random Forest Regression  

3. **Data Visualization**  
   - Exploratory Data Analysis (EDA) using Seaborn and Matplotlib.  
   - Graphical insights into how features correlate with insurance charges.

4. **Pre-Trained Model**  
   - A tuned Random Forest model (`rf_tuned.pkl`) is provided for deployment.

---

## 3. Project Structure

A typical layout of the repository might look like this:

```
├── sampleImages/
├── static/
│   ├── css/
│   │   └── [Your CSS Files]
│   ├── js/
│       └── [Your JavaScript Files]
├── templates/
│   └── Health_Insurance_Price_Prediction.html
├── Health_Insurance_Price_Prediction_Report.pdf
├── Medical Cost Insurance.ipynb
├── app.py
├── insurance.csv
├── rf_tuned.pkl
├── Procfile           (if deploying on platforms like Heroku)
└── README.md          (this file)
```

- **sampleImages/**: (Optional) Contains screenshots or additional images for documentation.  
- **static/**: Holds static files like CSS and JS.  
- **templates/**: Flask HTML templates (the main interface is `Health_Insurance_Price_Prediction.html`).  
- **Medical Cost Insurance.ipynb**: Jupyter Notebook with EDA, model training, and evaluation.  
- **app.py**: Flask application script.  
- **insurance.csv**: Dataset used for model training and EDA.  
- **rf_tuned.pkl**: Serialized (pickled) Random Forest model for quick prediction in production.  
- **Health_Insurance_Price_Prediction_Report.pdf**: A detailed project report containing methodology, results, and findings.

---

## 4. Dataset Information
- **Name**: `insurance.csv`  
- **Source**: [OSF.io (https://osf.io/7u5gy)](https://osf.io/7u5gy)  
- **Description**:  
  This dataset includes the following columns:
  - `age`: Age of the individual (years).  
  - `sex`: Gender (`male`, `female`).  
  - `bmi`: Body Mass Index, a measure of body fat based on height and weight.  
  - `children`: Number of children/dependents covered by insurance.  
  - `smoker`: Smoking status (`yes`/`no`).  
  - `region`: Residential region in the US (`northeast`, `northwest`, `southeast`, `southwest`).  
  - `charges`: Final individual medical costs billed by health insurance.

> **Note**: Ensure you download `insurance.csv` from the provided OSF link or keep the file in the repository to replicate the experiments.

---

## 5. Technologies & Libraries Used

1. **Programming Languages**  
   - **Python** (core language for data analysis, machine learning, and backend)  
   - **HTML/CSS** (front-end structure and styling)  
   - **JavaScript** (front-end interactivity)

2. **Web Framework**  
   - **Flask**: For creating the web application and handling routes.

3. **Python Libraries**  
   - **NumPy**: Numerical operations and array manipulations  
   - **Pandas**: Data manipulation and analysis  
   - **Matplotlib & Seaborn**: Data visualization  
   - **scikit-learn**: Machine learning algorithms (Linear Regression, SVR, Ridge, Random Forest)  
   - **pickle**: For model serialization and deserialization  

4. **Deployment**  
   - **Heroku**, **Render**, or any platform of your choice (Procfile is included if you choose Heroku).

---

## 6. Setup & Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<your-username>/Medical_Insurance_cost_prediction.git
   cd Medical_Insurance_cost_prediction
   ```

2. **Create a Virtual Environment (Recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Make sure you have a `requirements.txt` file. Then run:  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Jupyter Notebook**  
   If you want to explore or retrain the model, open the notebook:  
   ```bash
   jupyter notebook Medical\ Cost\ Insurance.ipynb
   ```
   This notebook contains EDA, model building, and evaluation steps.

---

## 7. Usage

1. **Run the Flask App**  
   From the root directory, execute:
   ```bash
   python app.py
   ```
   By default, the application will run on `http://127.0.0.1:5000/` (or another port if specified).

2. **Access the Web Interface**  
   - Open your web browser and navigate to the local server address (e.g., `http://127.0.0.1:5000/`).  
   - Enter the required details (age, gender, BMI, number of children, smoking status, region).  
   - Click the **Predict** button to get your estimated insurance charge.

3. **Model Predictions**  
   - The application uses the **Random Forest** model (`rf_tuned.pkl`) by default for predictions.  
   - If you wish to switch to another model, you can modify the code in `app.py` to load a different pickle file.

---

## 8. Exploratory Data Analysis (EDA)

The **Jupyter Notebook** (`Medical Cost Insurance.ipynb`) walks through:
1. **Data Cleaning**  
   - Handling missing values (if any).  
   - Encoding categorical variables (sex, smoker, region).

2. **Statistical Analysis & Visualization**  
   - Distribution of age, BMI, charges.  
   - Correlation matrix (heatmap) to see which features are most correlated with `charges`.  
   - Box plots and scatter plots to understand relationships (e.g., smoker vs. charges, region vs. charges).

3. **Insights**  
   - **Smoking status** is a strong determinant of higher insurance costs.  
   - **Age** and **BMI** also show strong positive correlations with charges.

---

## 9. Model Building & Evaluation

Multiple regression algorithms were trained and compared:
1. **Linear Regression**  
   - Simple baseline model.  
   - Interpretable but may underfit complex relationships.

2. **Support Vector Regression (SVR)**  
   - Captures non-linear relationships with different kernels.  
   - May require careful hyperparameter tuning.

3. **Ridge Regression**  
   - Helps reduce overfitting with an L2 penalty.  
   - Useful when features are correlated.

4. **Random Forest Regression**  
   - An ensemble of Decision Trees.  
   - Handles non-linearities and interactions well.  
   - **Chosen as the best model** based on cross-validation and R² score.

### Evaluation Metrics
- **R-squared (R²)**: Proportion of variance explained by the model.  
- **RMSE** (Root Mean Squared Error): Measures the standard deviation of prediction errors.  
- **Cross-Validation**: Applied for robust performance measurement.

---

## 10. Deployment

- **Local Deployment**: Run `python app.py` to start the Flask server locally.  

---

## 11. Results & Insights

- **Highest Predictive Performance**: Random Forest Regression outperformed other models in terms of R² and RMSE.  
- **Important Features**:  
  1. **Smoking status** – major cost driver.  
  2. **Age** – older individuals generally incur higher charges.  
  3. **BMI** – individuals with higher BMI tend to have higher medical costs.  
- **Lesser Influence**: Gender and number of children had smaller effects compared to smoking, BMI, and age.

---

## 12. Future Scope

1. **Real-Time Data Integration**  
   - Incorporate wearable device data or real-time health metrics to update insurance cost predictions dynamically.

2. **More Advanced Models**  
   - Explore Gradient Boosting, XGBoost, or neural networks for potentially better accuracy.

3. **Geographic Expansion**  
   - Include more granular location data or international datasets to generalize the model further.

4. **Cost Benefit Analysis**  
   - Extend the project to include risk assessment and ROI calculations for insurance companies.

5. **Interpretability & Explainability**  
   - Integrate SHAP (SHapley Additive exPlanations) or LIME to provide deeper insights into how each feature impacts individual predictions.

---

## 13. References

- [OSF Dataset](https://osf.io/7u5gy)  
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)  
- [Flask Documentation](https://flask.palletsprojects.com/en/2.2.x/)  
- *Health_Insurance_Price_Prediction_Report.pdf* in this repository for detailed methodology.

---

## 14. License

This project is licensed under the [MIT License](LICENSE) - feel free to modify and distribute as per the license terms.

---

### Thank You!
We hope this repository and its documentation help you explore and implement **Health Insurance Cost Prediction** in your own projects. If you have any questions or suggestions, please feel free to open an issue or submit a pull request. Happy coding!
