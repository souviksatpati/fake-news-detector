# 📰 Fake News Detection ML Application

## Overview
This project implements a machine learning solution to classify news articles as 'Real' or 'Fake'. The solution is deployed as an interactive web application using Streamlit, allowing users to paste any text snippet for instant classification.


## Methodology
This is a standard text classification problem solved using the following NLP and ML pipeline:

1.  **Dataset:** Used the publicly available [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.
2.  **Preprocessing:** Text was cleaned by converting to lowercase, removing punctuation, removing stopwords (common words), and applying **Porter Stemming** to reduce words to their root form.
3.  **Feature Engineering (Vectorization):** The cleaned text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
4.  **Model:** A **Passive Aggressive Classifier (PAC)** was chosen for its efficiency and strong performance on streaming text data.

## Model Performance

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **99.47%** |
| **Model** | Passive Aggressive Classifier |
| **Vectorizer**| TF-IDF (Term Frequency-Inverse Document Frequency) |

 
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

The application will open automatically in browser at `http://localhost:8501`.

## Project Files
* `app.py`: The main Streamlit script that handles the web interface and prediction logic.
* `requirements.txt`: Lists all Python library dependencies.
* `fake_news_detector_model.pkl`: The trained Passive Aggressive Classifier model.
* `tfidf_vectorizer.pkl`: The fitted TF-IDF object, essential for preprocessing new data identically to the training data.
Limitations

The model depends on the quality of the training dataset.

It may not correctly classify newly emerging types of misinformation.

The model analyzes text patterns and does not independently verify facts.

Satire, opinions, and partially true articles may be difficult to classify.

The reported accuracy is based on the evaluation dataset and may differ on real-world data.

Therefore, the prediction should be considered a machine learning result and not a final fact-check.
---
