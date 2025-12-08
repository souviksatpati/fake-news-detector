# 📰 Fake News Detection ML Application

## Overview
This project implements a machine learning solution to classify news articles as 'Real' or 'Fake'. The solution is deployed as an interactive web application using Streamlit, allowing users to paste any text snippet for instant classification.

## Website
### [Click Here!](https://fake-news-detection-dee.streamlit.app/)

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

## How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/dipmanjumdar/Fake-News-Detection-ML-App.git
    cd Fake-News-Detection-ML-App
    ```
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

---
