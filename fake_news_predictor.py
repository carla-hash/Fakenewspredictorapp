'''
Fake News Predictor App - Fake News Predictor Page

This Streamlit page allows users to input a news article's title and text,
and predicts whether the article is real or fake using a pre-trained machine
learning model. The page provides a visual gauge to display the model's
confidence in its prediction.
'''

import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@st.cache_resource
def load_model_and_vectorizer():
    '''
    Loads the pre-trained machine learning model and the TF-IDF vectorizer
    from the 'models' directory. These are used for predicting whether
    a news article is real or fake, and was trained on a dataset of news articles.

    Returns:
        model: The pre-trained machine learning model for fake news prediction.
        vectorizer: The pre-trained TF-IDF vectorizer used for text preprocessing.
    '''

    # Load the pre-trained model and vectorizer using pickle
    with open('models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Return the loaded model and vectorizer
    return model, vectorizer

def clean_text(text):
    '''
    Preprocesses input text for fake news prediction.

    This function performs the following steps:
    - Converts the text to lowercase.
    - Tokenizes the text into individual words.
    - Removes all tokens that are not alphanumeric (removes punctuation).
    - Removes English stopwords (common words that do not contribute to meaning).

    Parameters:
        text (str): The input text (title or article body) to preprocess.

    Returns:
        str: The cleaned and preprocessed text, with tokens joined by spaces.
    '''

    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))
    # Convert text to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)

@st.cache_data
def create_gauge(fake_prob, real_prob):
    '''
    Creates a polar gauge chart to visually represent the prediction confidence.
    The chart is divided into three sections: Real, Unsure, and Fake.

    Parameters:
    - fake_prob: Probability of the news being fake (float)
    - real_prob: Probability of the news being real (float)

    Returns:
    - A matplotlib figure object representing the gauge chart.
    '''

    # Initialize a polar plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='polar')

    # Define colors and sections for the gauge
    colors = ['green', 'yellow', 'red']
    sections = [0, 0.4 * np.pi, 0.6 * np.pi, np.pi]  # Divide the semi-circle

    # Plot each section of the gauge
    for i in range(len(colors)):
        ax.bar(
            x=[sections[i]],  # Start of each section
            width=sections[i + 1] - sections[i],  # Width of each section
            height=1,  # Radius of the section
            bottom=2,  # Start 2 units from the center
            color=colors[i],  # Color of the section
            edgecolor='black',
            linewidth=1,
            align='edge',
        )

    # Add labels for Real, Fake, and Unsure sections
    plt.annotate('Real', xy=(0.2 * np.pi, 2.5), rotation=-54, color='black', fontweight='bold', fontsize=8, ha='center', va='center')
    plt.annotate('Fake', xy=(0.8 * np.pi, 2.5), rotation=54, color='black', fontweight='bold', fontsize=8, ha='center', va='center')
    plt.annotate('Unsure', xy=(0.5 * np.pi, 2.5), rotation=0, color='black', fontweight='bold', fontsize=8, ha='center', va='center')

    # Add percentage markers
    plt.annotate('0%', xy=(np.pi, 3.1), color='black', fontsize=6, ha='right', va='center', annotation_clip=False)
    plt.annotate('40%', xy=(0.6 * np.pi, 3.1), rotation=10, color='black', fontsize=6, ha='center', va='center', annotation_clip=False)
    plt.annotate('60%', xy=(0.4 * np.pi, 3.1), rotation=-10, color='black', fontsize=6, ha='center', va='center', annotation_clip=False)
    plt.annotate('100%', xy=(0, 3.1), color='black', fontsize=6, ha='left', va='center', annotation_clip=False)

    # Add a needle-style arrow to indicate the prediction confidence
    plt.annotate(
        f'{real_prob:.2%}', xytext=(0, 0),
        xy=(fake_prob * np.pi, 2.25),
        arrowprops=dict(arrowstyle='wedge, tail_width=0.5', color='black', shrinkA=0),
        bbox=dict(boxstyle='circle', facecolor='black', linewidth=2.0),
        fontsize=6, color='white', ha='center'
    )

    # Hide the axis for a cleaner look
    ax.set_axis_off()

    # Return the figure object
    return fig

def main():
    '''
    Main function to render the Fake News Predictor page.
    This page allows users to input a news article's title and text,
    and predicts whether the article is real or fake.
    '''

    # Display the page title and descriptions
    st.title('Fake News Predictor')
    st.write('''
        This page allows you to predict whether a news article is real or fake. Enter the title and text of a news
        article to get started.
    ''')

    with st.container(border=True):
        # Input fields for the news article's title and text
        title = st.text_input('Enter the title of the news article:')
        text = st.text_area('Enter the text of the news article:')

        # Load the trained model and vectorizer
        model, vectorizer = load_model_and_vectorizer()

        # Predict button logic
        if st.button('Predict'):
            # Check if either fields are empty
            if title.strip() == '' or text.strip() == '':
                st.error('Please enter both a title or text to make a prediction.')
            else:
                # Preprocess title and text just like in training
                cleaned_title = clean_text(title)
                cleaned_text = clean_text(text)

                # Combine cleaned title and text for prediction
                combined_input = f'{cleaned_title} {cleaned_text}'

                # Vectorize the combined input using the pre-trained vectorizer
                input_vectorized = vectorizer.transform([combined_input])
                
                # Get prediction probabilities using the pre-trained model, and extract the probabilities
                probabilities = model.predict_proba(input_vectorized)[0]
                fake_prob = probabilities[0]  # Probability of being fake
                real_prob = probabilities[1]  # Probability of being real

                # Display the prediction results to the user
                st.subheader('Prediction Result', divider='gray')

                # Display a prediction message based on the probabilities
                if fake_prob > 0.6:
                    st.error('The model predicts this article is likely **Fake News**.')
                elif real_prob > 0.6:
                    st.success('The model predicts this article is likely **Real News**.')
                else:
                    st.warning('The model is unsure about the classification. Try to provide more news article text to improve the prediction.')

                # Plot and display the prediction gauge
                st.subheader('Prediction Gauge', divider='gray')

                # Add an expander to explain the gauge
                with st.expander('View explanation', expanded=True):
                    st.write('The gauge below shows the model\'s confidence in its prediction, displaying the probability of the article being real.')
                    st.write('The gauge is divided into three sections:')
                    st.write('''
                        - **Real**: The model predicts the article is real.
                        - **Fake**: The model predicts the article is fake.
                        - **Unsure**: The model is unsure about the classification.
                    ''')
                
                # Create and display the gauge chart
                gauge_fig = create_gauge(fake_prob, real_prob)
                st.pyplot(gauge_fig, use_container_width=False)

    # Add a decorative horizontal line at the bottom of the page
    st.markdown('---')

# Entry point of the script
if __name__ == '__main__':
    main()
