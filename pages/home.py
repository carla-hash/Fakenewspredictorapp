'''
Fake News Predictor App - Home Page

This Streamlit page introduces the app, explains its purpose, and provides an overview
of its features. It also displays team members and data sources, and encourages users
to explore the app using the navigation menu.
'''

import streamlit as st

def main():
    '''
    Main function to render the home page of the Fake News Detection App.
    This page introduces the app, explains its purpose, and provides an overview
    of its features. It also includes a call to action for users to explore the app.
    '''
    
    # Display the main title and header of the app
    st.title('📰 Fake News Detection App 📰')
    st.header('Welcome to the Fake News Detection App!')

    # Provide a brief introduction to the problem of fake news and the app's purpose
    st.write('''
        Welcome to the murky world of fake news, where facts are distorted and
        the truth is often just a disguise. In an age where misinformation
        spreads faster than ever, knowing what is true and what is not has
        become a daily challenge. Fake news is designed to manipulate, shared to
        divide, and intended to deceive, shock, and influence. By eroding trust
        in the media, damaging reputations, and fueling confusion or fear, it
        undermines the very foundations of our society.
    ''')

    # Explain how the app helps users identify fake news
    st.write('''
        That is where this platform comes in to help you uncover the truth about
        any topic or paragraph. This site takes you to the heart of the
        disinformation machine. Its mission: to give you the tools you need to
        distinguish truth from fiction and never be misled again.
    ''')
    
    # Highlight the key features of the app with icons for better visualization
    st.markdown('''
        ### Features:
        - 📊 **Data Visualization**: Explore the dataset and visualize patterns
            in fake and real news.
        - 🔍 **News Search**: Search for news articles using various filters and
            advanced search options.
        - 🤖 **Fake News Prediction**: Enter a news article's title and text to
            predict whether it is real or fake.
    ''')
    
    # Encourage users to explore the app's features using the navigation menu
    st.write('''
        Ready to dive in? Use the navigation menu on the left to explore the
        different features of the app. Whether you want to visualize the
        differences between real and fake news, search for articles based on
        keywords, or predict the search for news articles, we have you covered!
    ''')

    # Add a decorative horizontal line
    st.markdown('---')

    # Add a caption for the team members
    st.caption('''
        Team Members: Alicia Sanna, Davide Minonne, Carla Lea Schmitt, Philippe Sensi
    ''')

    # Add a caption for the sources of the datasets
    st.caption('''
        This app uses fake news datasets sourced from multiple sources.
        - Kaggle Fake News Dataset #1: https://www.kaggle.com/datasets/antonioskokiantonis/newscsv
        - Kaggle Fake News Dataset #2: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data
        - Fake News Corpus: https://github.com/several27/FakeNewsCorpus (only a subset of the dataset is used)
    ''')

    # Add a caption for the NewsAPI source
    st.caption('''
        The app uses NewsAPI for fetching news articles in the News Search page. For more information,
        visit: https://newsapi.org/
    ''')

# Entry point of the script
if __name__ == '__main__':
    main()
