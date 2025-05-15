'''
Fake News Predictor App - Main Application

This is the main entry point for the Streamlit application. It sets up the navigation
between the Home, Dataset Dashboard, News Search, and Fake News Predictor pages,
and ensures that required NLTK resources are downloaded for text processing.
'''

import streamlit as st
import nltk

@st.cache_resource
def download_nltk_resources():
    '''
    Downloads necessary NLTK resources for text processing.
    This function is cached to avoid redundant downloads.
    '''

    # Download stopwords for text preprocessing
    nltk.download('stopwords')
    # Download punkt tokenizer for sentence splitting
    nltk.download('punkt')
    # Download punkt_tab (if required for punkt tokenizer)
    nltk.download('punkt_tab')

def main():
    '''
    The main function that initializes and runs the Streamlit app.
    It sets up navigation between different pages and downloads required
    resources.
    '''

    # Define each of the pages for the Streamlit app
    home_page = st.Page('pages/home.py', title='Home')
    dataset_visualization_page = st.Page('pages/dataset_dashboard.py', title='Dataset Dashboard')
    news_search_page = st.Page('pages/news_search.py', title='News Search')
    fake_news_predictor_page = st.Page('pages/fake_news_predictor.py', title='Fake News Predictor')

    # Create a navigation menu for the app
    pages = st.navigation([
        home_page,
        dataset_visualization_page,
        news_search_page,
        fake_news_predictor_page
    ])

    # Set the layout of the Streamlit app to wide by default
    st.set_page_config(layout='wide')

    # Run the selected page from the navigation menu
    pages.run()

    # Download NLTK resources required for text processing
    download_nltk_resources()

# Entry point of the application
if __name__ == '__main__':
    main()
