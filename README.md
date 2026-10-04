# 📰 AI-Powered Fake News Detection System

An end-to-end Machine Learning and Natural Language Processing web application that analyzes news headlines and articles and predicts whether the given content is **Real** or **Fake**.

Built using a **Passive Aggressive Classifier (PAC)**, **TF-IDF Vectorization**, **NLTK**, and an interactive **Streamlit** interface.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [How the System Works](#-how-the-system-works)
- [NLP Pipeline](#-nlp-pipeline)
- [Machine Learning Model](#-machine-learning-model)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Application Usage](#-application-usage)
- [Example Workflow](#-example-workflow)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 📖 Overview

Fake news and misinformation can spread rapidly through digital media platforms and online news sources.

The **AI-Powered Fake News Detection System** demonstrates how Machine Learning and Natural Language Processing can be used to classify news content based on patterns learned from a labelled dataset.

The application accepts news text directly from the user or can extract article content from a URL. The extracted or entered text is processed through an NLP pipeline and passed to a trained Machine Learning model.

The system then produces a prediction:

- **REAL** — The model predicts that the content belongs to the real-news class.
- **FAKE** — The model predicts that the content belongs to the fake-news class.

This project is primarily intended for educational and demonstration purposes.

---

## 🎯 Core Objectives

The main objectives of this project are:

- Apply Machine Learning to a real-world text classification problem.
- Demonstrate Natural Language Processing techniques.
- Process and normalize unstructured news text.
- Convert text into numerical features using TF-IDF.
- Use a Passive Aggressive Classifier for binary classification.
- Provide a simple interface for testing news content.
- Demonstrate how a trained ML model can be integrated into a web application.
- Provide fast predictions for submitted news text.

---

## ✨ Key Features

### 📝 Direct Text Verification

Users can enter:

- News headlines
- News snippets
- News paragraphs
- Full news articles

The system processes the provided text and generates a prediction.

### ⚡ Sample News

The application provides sample input options so users can quickly test the prediction system without manually entering an article.

### 🔗 URL-Based Analysis

Users can provide an online news article URL.

The application attempts to:

1. Fetch the webpage.
2. Extract visible article text.
3. Process the extracted content.
4. Pass the processed text to the trained model.
5. Display the prediction.

### 🎨 Modern User Interface

The Streamlit interface includes a customized modern design with:

- Dark interface
- Glassmorphic cards
- Custom typography
- Responsive layout
- Prediction badges
- Interactive buttons
- Content statistics
- Pipeline information

### 📊 Content Statistics

The application can display information such as:

- Word count
- Character count
- Prediction result
- Model information
- Processing/pipeline information

---

## 🧠 How the System Works

The complete workflow can be summarized as:

```text
User Input
    │
    ├─────────────── Direct Text
    │
    └─────────────── News URL
                         │
                         ▼
                  Web Page Scraping
                  BeautifulSoup
                         │
                         ▼
                   Article Text
                         │
                         ▼
                Text Preprocessing
                         │
                         ▼
                Tokenization
                         │
                         ▼
                Stopword Removal
                         │
                         ▼
                   Stemming
                         │
                         ▼
                TF-IDF Vectorization
                         │
                         ▼
          Passive Aggressive Classifier
                         │
                         ▼
                    Prediction
                    /       \
                   /         \
               REAL          FAKE
