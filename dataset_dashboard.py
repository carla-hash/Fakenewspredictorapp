'''
Fake News Predictor App - Dataset Dashboard Page

This Streamlit page provides interactive data exploration and visualization tools
for analyzing fake and real news datasets. Users can explore the data, filter articles,
and view visual insights such as word clouds, top words, and length distributions.
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter

@st.cache_data
def load_data():
    '''
    Load and preprocess the dataset from multiple CSV files.

    This function reads four datasets containing news articles, assigns labels to indicate whether 
    the articles are fake or real, removes unnecessary columns, combines them into a single DataFrame, 
    and applies data cleaning steps such as removing empty rows and limiting the text length.

    Returns:
        pd.DataFrame: A combined and cleaned DataFrame containing all the datasets.
    '''

    # Load the first dataset containing fake news articles
    fake_news_df = pd.read_csv('data/Fake.csv')
    # Load the second dataset containing real news articles
    real_news_df = pd.read_csv('data/True.csv')
    # Load the third dataset containing a mix of fake and real news articles
    more_news_df = pd.read_csv('data/news.csv')
    # Load the fourth dataset containing a larger sample of news articles
    large_news_df = pd.read_csv('data/news_fakenewscorpus.csv')

    # Add a 'label' column to the fake news dataset and drop unnecessary columns
    fake_news_df['label'] = 'fake'
    fake_news_df = fake_news_df.drop(['subject', 'date'], axis=1)

    # Add a 'label' column to the real news dataset and drop unnecessary columns
    real_news_df['label'] = 'real'
    real_news_df = real_news_df.drop(['subject', 'date'], axis=1)

    # Standardize the labels in the third dataset and drop unnecessary columns
    more_news_df['label'] = more_news_df['label'].replace({'FAKE': 'fake', 'REAL': 'real'})
    more_news_df = more_news_df.drop(['Unnamed: 0'], axis=1)

    # Combine all datasets into a single DataFrame
    combined_df = pd.concat([fake_news_df, real_news_df, more_news_df, large_news_df], ignore_index=True)

    # Remove rows where the 'text' field is empty or contains only whitespace
    combined_df = combined_df[~(combined_df['text'].str.strip() == '')]

    # Remove rows where the 'title' field is empty or contains only whitespace
    combined_df = combined_df[~(combined_df['title'].str.strip() == '')]

    # Remove rows where the length of the 'text' field exceeds 20,000 characters
    combined_df = combined_df[combined_df['text'].str.len() <= 20000]

    # Return the cleaned and combined dataset
    return combined_df

@st.cache_data
def preprocess_data():
    '''
    Load a preprocessed version of the dataset.

    This function reads a preprocessed dataset from a CSV file. The preprocessed dataset
    has already undergone cleaning and transformation steps, making it ready for analysis.
    See machine_learning_model.ipynb for the preprocessing steps.

    Returns:
        pd.DataFrame: A preprocessed DataFrame loaded from the specified CSV file.
    '''

    # Load the preprocessed dataset from a CSV file
    # The file is expected to contain cleaned and transformed data
    preprocessed_df = pd.read_csv('data/news_preprocessed.csv')

    # Return the preprocessed dataset
    return preprocessed_df

@st.cache_data
def create_wordclouds(data):
    '''
    Generate and display word clouds for fake and real news titles and text.

    This function creates word clouds to visualize the most frequent words in the titles
    and text of fake and real news articles. The size of each word in the word cloud
    represents its frequency in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset containing 'title', 'text', and 'label' columns.
                             The 'label' column should indicate whether an article is 'fake' or 'real'.
    '''

    # Filter the dataset to extract titles and text for fake and real news
    fake_titles = data[data['label'] == 'fake']['title']  # Titles of fake news articles
    real_titles = data[data['label'] == 'real']['title']  # Titles of real news articles
    fake_text = data[data['label'] == 'fake']['text']    # Text of fake news articles
    real_text = data[data['label'] == 'real']['text']    # Text of real news articles

    # Generate word clouds for fake and real news titles
    fake_titles_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_titles))
    real_titles_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(real_titles))

    # Generate word clouds for fake and real news text
    fake_text_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_text))
    real_text_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(real_text))

    # Create a 2x2 grid of subplots to display the word clouds
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Display the word cloud for fake news titles
    axes[0, 0].imshow(fake_titles_wordcloud, interpolation='bilinear')
    axes[0, 0].axis('off')  # Remove axes for better visualization
    axes[0, 0].set_title('Word Cloud for Fake News Titles', fontsize=16)

    # Display the word cloud for real news titles
    axes[0, 1].imshow(real_titles_wordcloud, interpolation='bilinear')
    axes[0, 1].axis('off')  # Remove axes for better visualization
    axes[0, 1].set_title('Word Cloud for Real News Titles', fontsize=16)

    # Display the word cloud for fake news text
    axes[1, 0].imshow(fake_text_wordcloud, interpolation='bilinear')
    axes[1, 0].axis('off')  # Remove axes for better visualization
    axes[1, 0].set_title('Word Cloud for Fake News Text', fontsize=16)

    # Display the word cloud for real news text
    axes[1, 1].imshow(real_text_wordcloud, interpolation='bilinear')
    axes[1, 1].axis('off')  # Remove axes for better visualization
    axes[1, 1].set_title('Word Cloud for Real News Text', fontsize=16)

    # Adjust the layout to prevent overlapping and display the word clouds
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def create_pie_chart(df):
    '''
    Generate and display a pie chart showing the distribution of fake and real news articles.

    This function calculates the proportion of fake and real news articles in the dataset
    and visualizes it as a pie chart. The chart uses distinct colors for each label and
    displays the percentage of each category.

    Paramters:
        df (pd.DataFrame): The dataset containing a 'label' column with values 'fake' or 'real'.
    '''

    # Count the number of fake and real news articles
    label_counts = df['label'].value_counts()

    # Create a pie chart to visualize the label distribution
    fig, ax = plt.subplots()
    ax.pie(
        label_counts,  # Data for the pie chart
        labels=['fake', 'real'],  # Labels for the categories
        autopct='%1.1f%%',  # Format for displaying percentages
        startangle=90,  # Start the pie chart at 90 degrees
        colors=['#ef8a62', '#67a9cf']  # Colors for fake and real news
    )

    # Ensure the pie chart is circular by setting an equal aspect ratio
    ax.axis('equal')

    # Display the pie chart in the Streamlit app
    st.pyplot(fig)

@st.cache_data
def create_bar_chart(df):
    '''
    Generate and display a bar chart showing the count of fake and real news articles.

    This function calculates the number of fake and real news articles in the dataset
    and visualizes it as a bar chart. The chart uses distinct colors for each label
    and includes gridlines for better readability.

    Parameters:
        df (pd.DataFrame): The dataset containing a 'label' column with values 'fake' or 'real'.
    '''

    # Count the number of fake and real news articles
    label_counts = df['label'].value_counts()

    # Create a bar chart to visualize the label distribution
    fig, ax = plt.subplots()
    ax.bar(
        label_counts.index,  # Categories ('fake' and 'real')
        label_counts.values,  # Counts for each category
        color=['#ef8a62', '#67a9cf']  # Colors for fake and real news
    )

    # Add gridlines to the y-axis for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add labels to the axes
    ax.set_ylabel('Count')  # Label for the y-axis
    ax.set_xlabel('Label')  # Label for the x-axis

    # Display the bar chart in the Streamlit app
    st.pyplot(fig)

@st.cache_data
def create_length_histograms(df):
    '''
    Generate and display histograms showing the distribution of title and text lengths
    for fake and real news articles.

    This function calculates the lengths of titles and text for each article in the dataset,
    groups them by their labels ('fake' or 'real'), and visualizes the distributions as histograms.
    It also overlays the mean length for each label as a vertical dashed line.

    Parameters:
        df (pd.DataFrame): The dataset containing 'title', 'text', and 'label' columns.
    '''

    # Calculate the length of each title and text
    df['title_len'] = df['title'].astype(str).apply(len)  # Length of titles
    df['text_len'] = df['text'].astype(str).apply(len)    # Length of text

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # Title Length Histogram
    for label in df['label'].unique():
        # Filter the dataset by label (fake or real)
        subset = df[df['label'] == label]
        # Calculate the mean title length for the current label
        mean_title_len = subset['title_len'].mean()
        # Plot the histogram for title lengths
        axes[0].hist(
            subset['title_len'], bins=50, alpha=0.5,
            label=f'{label} (mean={mean_title_len:.1f})',
            color='#67a9cf' if label == 'real' else '#ef8a62', edgecolor='black'
        )
        # Add a vertical dashed line for the mean title length
        axes[0].axvline(
            mean_title_len, color='#67a9cf' if label == 'real' else '#ef8a62',
            linestyle='--', linewidth=1.5
        )

    # Customize the title length histogram
    axes[0].set_title('Title Length Histogram by Label')
    axes[0].set_xlabel('Title Length')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Label')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Text Length Histogram
    for label in df['label'].unique():
        # Filter the dataset by label (fake or real)
        subset = df[df['label'] == label]
        # Calculate the mean text length for the current label
        mean_text_len = subset['text_len'].mean()
        # Plot the histogram for text lengths
        axes[1].hist(
            subset['text_len'], bins=50, alpha=0.5,
            label=f'{label} (mean={mean_text_len:.1f})',
            color='#67a9cf' if label == 'real' else '#ef8a62', edgecolor='black'
        )
        # Add a vertical dashed line for the mean text length
        axes[1].axvline(
            mean_text_len, color='#67a9cf' if label == 'real' else '#ef8a62',
            linestyle='--', linewidth=1.5
        )

    # Customize the text length histogram
    axes[1].set_title('Text Length Histogram by Label')
    axes[1].set_xlabel('Text Length')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Label')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Adjust the layout to prevent overlapping and display the histograms
    plt.tight_layout()
    st.pyplot(fig)

def main():
    '''
    Main function to render the Streamlit dashboard for the Fake News Dataset.

    This function organizes the dashboard into three tabs:
    1. Dataset Overview: Provides a summary of the dataset, including statistics and visualizations.
    2. Dataset Exploration: Allows users to filter and explore the dataset interactively.
    3. Visualizations: Displays advanced visualizations such as word clouds and length histograms.

    The dashboard is designed to help users understand the dataset and gain insights into the distribution
    and characteristics of fake and real news articles.
    '''

    # Set the title and introductory text for the dashboard
    st.title('Fake News Dataset Dashboard')
    st.write('This dashboard provides an overview of the dataset, including data information, visualizations, and insights.')
    st.write('Use the tabs below to navigate through the content.')

    # Load the dataset
    data = load_data()

    # Randomize the dataset for better representation
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create tabs for organizing the dashboard content
    tab1, tab2, tab3 = st.tabs(['Dataset Overview', 'Dataset Exploration', 'Visualizations'])

    # Tab 1: Dataset Overview
    with tab1:
        # Display the dataset overview header and description
        st.header('Dataset Overview')
        st.write('''
            This application uses multiple fake news datasets to build a machine
            learning model that predicts whether a news article is fake or real.
        ''')
        st.write('''
            The dataset contains thousands of news articles labeled as either
            fake or real, sourced from a mix of credible news organizations and
            websites known for publishing unreliable content.
        ''')

        # Display dataset statistics in a grid layout
        col1, col2, col3 = st.columns(3, border=True)

        with col1:
            # Display the total number of articles
            st.metric(label='Total Articles', value=len(data))

        with col2:
            # Display the number of fake articles
            fake_count = len(data[data['label'] == 'fake'])
            st.metric(label='Fake Articles', value=fake_count)

        with col3:
            # Display the number of real articles
            real_count = len(data[data['label'] == 'real'])
            st.metric(label='Real Articles', value=real_count)

        # Display visualizations for label distribution
        col4, col5 = st.columns(2, border=True)
        with col4:
            st.subheader('Label Distribution')
            create_pie_chart(data)  # Generate a pie chart for label distribution

        with col5:
            st.subheader('Bar Chart of News Labels')
            create_bar_chart(data)  # Generate a bar chart for label distribution

    # Tab 2: Dataset Exploration
    with tab2:
        # Display the dataset exploration header and description
        st.header('Dataset Exploration')
        st.write('This section allows you to explore the dataset in more detail.')

        with st.container(border=True):
            # Display the dataset table header and description
            st.subheader('Dataset Table', divider='gray')
            st.write('The table below displays the fake news dataset. You can filter the dataset based on the label and search for specific words in the article text. The table is limited to 1000 rows for better performance.')

            # Dataset column descriptions
            with st.expander('View Dataset Column Descriptions'):
                st.write('The dataset contains the following columns:')
                st.write('''
                    - **title**: The title of the news article.
                    - **text**: The content of the news article.
                    - **label**: The label indicating whether the article is fake or real.
                ''')

            # Form for filtering the dataset
            with st.form('filter_form', enter_to_submit=False):
                st.write('Below you can filter the dataset based on the label and search for specific words in the article text.')

                # Filter options
                label_filter = st.selectbox('Select news type:', options=['All', 'Fake', 'Real'])
                search_words = st.text_input('Search for words in the news text (separate multiple words with commas):')
                search_operator = st.radio('Search operator:', options=['OR', 'AND'], index=0, horizontal=True)

                # Explanation of the search operator
                with st.expander('View search operator explanation'):
                    st.write('**Search Operator:** Choose how to combine multiple search words:')
                    st.write('- **OR**: Articles containing **any** of the words will be included.')
                    st.write('- **AND**: Only articles containing **all** of the words will be included.')

                # Submit button for applying filters
                submit_button = st.form_submit_button('Apply Filters')

            # Apply filters to the dataset
            filtered_data = data.copy()

            # Filter by label
            if label_filter == 'Fake':
                filtered_data = filtered_data[filtered_data['label'] == 'fake']
            elif label_filter == 'Real':
                filtered_data = filtered_data[filtered_data['label'] == 'real']

            # Filter by search words
            if search_words.strip():
                # Split the input into a list of words and strip whitespace
                search_words_list = [word.strip() for word in search_words.split(',') if word.strip()]

                if search_operator == 'OR':
                    # Use 'OR' operator: Match any of the words
                    filtered_data = filtered_data[
                        filtered_data['text'].str.contains('|'.join(search_words_list), case=False, na=False)
                    ]
                elif search_operator == 'AND':
                    # Use 'AND' operator: Match all of the words
                    for word in search_words_list:
                        filtered_data = filtered_data[
                            filtered_data['text'].str.contains(word, case=False, na=False)
                        ]

            # Display the number of results after filtering
            total_results = len(filtered_data)
            st.write(f'**Number of results after filtering:** {total_results}')

            # Limit the displayed dataset to 1000 rows for performance
            if total_results > 1000:
                st.warning('The table has been limited to 1000 results for better performance.')
                filtered_data = filtered_data.sample(n=1000, random_state=42)
                filtered_data = filtered_data.sort_index()  # Maintain the original order

            # Display the filtered dataset
            st.dataframe(filtered_data)

    # Tab 3: Visualizations
    with tab3:
        # Preprocess the dataset for visualizations
        data = preprocess_data()

        # Display the visualizations header and description
        st.header('Visualizations')
        st.write('This section provides advanced visualizations to help you understand the dataset better.')

        # Filter data for visualizations
        with st.popover('Filter Data for Visualizations'):
            st.subheader('Filter Data for Visualizations', divider='gray')
            st.write('Use the form below to filter the dataset before generating visualizations.')

            with st.form('visualization_filter_form', enter_to_submit=False):
                # Filter options
                search_words = st.text_input('Search for words in the news text (separate multiple words with commas):')
                search_operator = st.radio('Search operator:', options=['OR', 'AND'], index=0, horizontal=True)

                # Explanation of the search operator
                with st.expander('View search operator explanation'):
                    st.write('**Search Operator:** Choose how to combine multiple search words:')
                    st.write('- **OR**: Articles containing **any** of the words will be included.')
                    st.write('- **AND**: Only articles containing **all** of the words will be included.')

                # Submit button for applying filters
                submit_button = st.form_submit_button('Apply Filters')

            # Apply filters to the dataset
            filtered_data_2 = data.copy()

            # Filter by search words
            if search_words.strip():
                # Split the input into a list of words and strip whitespace
                search_words_list = [word.strip() for word in search_words.split(',') if word.strip()]

                if search_operator == 'OR':
                    # Use 'OR' operator: Match any of the words
                    filtered_data_2 = filtered_data_2[
                        filtered_data_2['text'].str.contains('|'.join(search_words_list), case=False, na=False)
                    ]
                elif search_operator == 'AND':
                    # Use 'AND' operator: Match all of the words
                    for word in search_words_list:
                        filtered_data_2 = filtered_data_2[
                            filtered_data_2['text'].str.contains(word, case=False, na=False)
                        ]

            # Display the number of results after filtering
            total_results = len(filtered_data_2)
            st.write(f'**Number of results after filtering:** {total_results}')

        # If no results are found, display a message
        if total_results == 0:
            st.warning('No results found for the specified search criteria.')
            return
        
        # If results are found, proceed with visualizations

        # Word Clouds
        with st.container(border=True):
            st.subheader('Word Clouds', divider='gray')
            with st.expander('View explanation'):
                st.write('Word clouds visualize the most frequent words in the text data. The size of each word indicates its frequency in the dataset. Larger words appear more frequently in the text.')
                st.write('The word clouds below show the most common words in fake and real news titles and text, respectively.')

            # Generate and display word clouds
            create_wordclouds(filtered_data_2)

        # Length Histograms
        with st.container(border=True):
            st.subheader('Length Histograms', divider='gray')
            with st.expander('View explanation'):
                st.write('The histograms below show the distribution of text lengths for both fake and real news articles. The x-axis represents the length of the title or text, while the y-axis shows the count of articles with that length.')

            # Generate and display length histograms
            create_length_histograms(filtered_data_2)

    # Add a decorative horizontal line at the bottom of the page
    st.markdown('---')

# Entry point of the script
if __name__ == '__main__':
    main()
