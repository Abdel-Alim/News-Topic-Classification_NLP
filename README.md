# 📰 News Topic Classification using Machine Learning

A Natural Language Processing (NLP) project that classifies news articles into four categories using machine learning algorithms and TF-IDF text representation.

## 📌 Project Overview

This project focuses on automatically identifying the topic of a news article based on its textual content.

The project uses the AG News dataset, applies text cleaning and preprocessing techniques, converts text into numerical features using TF-IDF, and trains multiple machine learning models to perform multi-class text classification.

The trained model is evaluated using classification metrics and a confusion matrix, with additional analysis of misclassified articles.

## 🎯 Project Objectives

* Build a machine learning model for news topic classification.
* Explore the distribution and length of news articles.
* Apply text cleaning and NLP preprocessing techniques.
* Convert text into numerical features using TF-IDF.
* Compare different machine learning algorithms.
* Evaluate model performance and analyze classification errors.
* Save the trained model and vectorizer for future use.

## 📂 Dataset

**Dataset:** AG News

The project loads the dataset directly using the Hugging Face `datasets` library.

The dataset contains news articles belonging to four categories:

| Label | Category |
| ----- | -------- |
| 0     | World    |
| 1     | Sports   |
| 2     | Business |
| 3     | Sci/Tech |

The notebook uses the predefined training and test splits provided by the dataset.

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* Hugging Face Datasets
* Joblib
* Jupyter Notebook

## 🔄 Project Workflow

### 1. Data Loading and Exploration

* Load the AG News dataset.
* Convert the training and test datasets into Pandas DataFrames.
* Map numerical labels to readable category names.
* Examine dataset dimensions.
* Visualize class distribution and article length distribution.

### 2. Text Cleaning

The `clean_text()` function performs the following operations:

* Converts text to lowercase.
* Removes URLs.
* Removes HTML tags.
* Removes non-alphabetic characters.
* Normalizes whitespace.

### 3. NLP Preprocessing

The `preprocess()` function applies:

* Tokenization using whitespace splitting.
* English stopword removal.
* Word lemmatization using NLTK's `WordNetLemmatizer`.

These steps help prepare the text for feature extraction.

### 4. Feature Extraction

The project uses `TfidfVectorizer` to convert processed text into numerical features.

Configuration:

| Parameter          | Value                |
| ------------------ | -------------------- |
| Maximum features   | 20,000               |
| N-gram range       | Unigrams and bigrams |
| Feature extraction | TF-IDF               |

The training data is split into training and validation sets using an 80/20 split, with stratification and a fixed random state of 42.

### 5. Model Training

Three machine learning algorithms are trained and evaluated:

| Model                                      | Role                          |
| ------------------------------------------ | ----------------------------- |
| Multinomial Naive Bayes                    | Baseline model                |
| Logistic Regression                        | Advanced classification model |
| Linear Support Vector Machine (Linear SVM) | Selected model                |

The Linear SVM model is assigned as the best model in the notebook and is used for subsequent evaluation and saving.

### 6. Model Evaluation

The project evaluates classification performance using:

* Validation accuracy.
* Classification report, including precision, recall, and F1-score.
* Confusion matrix.
* Final test accuracy.

The notebook also investigates misclassified articles to identify common classification errors and challenging category pairs.

### 7. Model Saving

The trained model and TF-IDF vectorizer are saved using Joblib:

```python
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

These artifacts can be reused for future predictions without retraining the model.

## 📊 Results & Evaluation

The notebook compares three machine learning algorithms and selects Linear SVM for the final evaluation.

The evaluation includes validation and test accuracy, a classification report, and a confusion matrix.

**Performance metrics:** The exact numerical results are generated when the notebook is executed. No specific accuracy score is reported here because it has not been verified.

## 📁 Project Structure

```text
News-Topic-Classification/
│
├── News Topic Classification .ipynb
│
├── model.pkl
│
├── vectorizer.pkl
│
└── README.md
```

**Note:** The model and vectorizer files are generated when the notebook is executed. They may not be present in the repository until you save and upload them.

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn nltk datasets scikit-learn joblib jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook:

`News Topic Classification .ipynb`

### 4. Run the notebook

Execute the notebook cells sequentially to:

1. Load and explore the dataset.
2. Clean and preprocess the text.
3. Extract TF-IDF features.
4. Train and evaluate the models.
5. Analyze classification errors.
6. Save the trained model and vectorizer.

The notebook downloads the required NLTK resources (`stopwords` and `wordnet`) during execution.

## 🚀 Future Improvements

* Tune model hyperparameters using cross-validation.
* Compare additional algorithms and NLP techniques.
* Improve preprocessing and investigate the impact of different text-cleaning strategies.
* Analyze the most frequently confused news categories.
* Build a reusable prediction pipeline that includes text preprocessing and vectorization.
* Develop a simple web application for interactive news classification.
* Experiment with transformer-based models such as BERT.

## 👨‍💻 Author

**Abdel-Alim Wagih Fathy**

 Data Analyst | Machine Learning & NLP Enthusiast

GitHub: [Abdel-Alim](https://github.com/Abdel-Alim)

---

⭐ If you find this project useful, feel free to star the repository!
