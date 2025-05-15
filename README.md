# Fake News Predictor App

This app is designed to help uncover the truth behind news articles using machine learning. In an age where misinformation spreads rapidly, this app empowers users to distinguish between real and fake news. The app provides tools for data visualization, news search, and fake news prediction to help users navigate in the world of information with greater confidence.

## Features

- 📊 **Data Visualization**: Explore the dataset and visualize patterns in fake and real news.
- 🔍 **News Search**: Search for news articles using various filters and advanced search options.
- 🤖 **Fake News Prediction**: Enter a news article's title and text to predict whether it is real or fake.

## Requirements

All required Python packages are listed in the `requirements.txt` file.

## Setup Instructions

1. **Clone the repository and change directory to the project**

    Clone the project repository by replacing `<github-repo-url>` with the main URL of the GitHub repository.

    ```bash
    git clone <github-repo-url>
    cd fake-news-predictor-app
    ```

2. **Create a virtual environment**

    Using **conda**:

    ```bash
    conda create --name "fake-news-predictor-app"
    conda activate fake-news-predictor-app
    ```

3. **Install dependencies**

    Using **conda**:

    ```bash
    conda install --file requirements.txt
    ```

4. **Get a NewsAPI key and configure secrets**

    Go to [NewsAPI](https://newsapi.org/) and sign up for a free API key.
    Open `.streamlit/secrets.toml` and replace `your_api_key_here` with the API key provided by NewsAPI.

    ```toml
    news_api_key = "your_api_key_here"
    ```

5. **Run the Streamlit app**

    ```bash
    streamlit run app.py
    ```

6. **Open app in your browser**

    Streamlit will automatically open a browser to access the app. If it doesn't, open the local URL (e.g. http://localhost:8501) provided by Streamlit in the browser manually.

## Project Structure

```
fake-news-predictor-app/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   ├── news.csv
│   ├── news_fakenewscorpus.csv
│   └── news_preprocessed.csv
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── pages/
│   ├── home.py
│   ├── dataset_dashboard.py
│   ├── news_search.py
│   └── fake_news_predictor.py
├── .streamlit/
│   └── secrets.toml
└── machine_learning_model.ipynb
```

## File & Directory Explanations

- **app.py**  
  Main entry point for the Streamlit application.

- **requirements.txt**  
  Lists all Python dependencies needed to run the project.

- **README.md**  
  Project documentation and setup instructions.

- **data/**  
  Folder containing datasets used for training and testing.
  - **Fake.csv**: Fake news articles.
  - **True.csv**: Real news articles.
  - **news.csv**: Additional news dataset containing both real and fake news.
  - **news_fakenewscorpus.csv**: Subset of FakeNewsCorpus dataset.
  - **news_preprocessed.csv**: Preprocessed dataset ready for visualizations.

- **models/**  
  Folder containing trained machine learning models and vectorizers.
  - **model.pkl**: Trained ML model for fake news prediction.
  - **vectorizer.pkl**: Fitted TfidfVectorizer vectorizer.

- **pages/**  
  Streamlit multipage app scripts.
  - **home.py**: Landing page for the app.
  - **dataset_dashboard.py**: Data exploration and visualization dashboard.
  - **news_search.py**: News search functionality using NewsAPI.
  - **fake_news_predictor.py**: Fake news prediction using trained machine learning model.

- **.streamlit/**  
  Streamlit configuration files.
  - **secrets.toml**: Secrets (such as API keys) for Streamlit.

- **machine_learning_model.ipynb**  
  Jupyter notebook for preprocessing the fake news data and model training and exporting.

## Dataset Sources

This app uses fake news datasets sourced from multiple sources:
- [Kaggle Fake News Dataset #1](https://www.kaggle.com/datasets/antonioskokiantonis/newscsv)
- [Kaggle Fake News Dataset #2](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)
- [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) (only a subset of the dataset is used)

## News API

The app uses [NewsAPI](https://newsapi.org/) for fetching news articles in the News Search page.

## Team Members

Alicia Sanna, Davide Minonne, Carla Lea Schmitt, Philippe Sensi
