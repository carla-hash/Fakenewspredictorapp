'''
Fake News Predictor App - News Search Page

This Streamlit page allows users to search for news articles using advanced query syntax
and various filters. It fetches articles from the NewsAPI and displays them with details
such as title, source, author, image, description, and publication date.
'''

import urllib.parse

from datetime import datetime
from dateutil.relativedelta import relativedelta

import requests
import streamlit as st

def fetch_news(api_key, query, from_date, to_date, sort_by):
    '''
    Fetches news articles from the NewsAPI based on the provided query and filters.

    Parameters:
        api_key (str): API key for authenticating with NewsAPI.
        query (str): Search query string.
        from_date (str): Start date for the search range (YYYY-MM-DD).
        to_date (str): End date for the search range (YYYY-MM-DD).
        sort_by (str): Sorting criteria (e.g., relevancy, popularity, publishedAt).

    Returns:
        list: A list of articles if the request is successful, otherwise an empty list.
    '''

    # URL-encode the query to ensure it is safe for use in the API URL
    encoded_query = urllib.parse.quote(query)

    # Construct the API URL with the provided parameters
    url = f'https://newsapi.org/v2/everything?q={encoded_query}&from={from_date}&to={to_date}&sortBy={sort_by}&apiKey={api_key}'

    # Send a GET request to the API
    response = requests.get(url)

    # Check if the response is successful
    if response.status_code == 200:
        # Return the list of articles from the response JSON
        return response.json().get('articles', [])
    else:
        # Display an error message in the Streamlit app if the request fails
        st.error('Error fetching news articles. Please check your API key and parameters.')
        return []

def main():
    '''
    Main function to render the News Search page in the Streamlit app.
    Allows users to search for news articles using advanced query syntax and filters.
    '''

    # Set the title and description of the page
    st.title('News Search')
    st.write('Search for news articles using various filters and advanced search options.')

    with st.container(border=True):
        # Section for entering the search query
        st.subheader('Search Query', divider='gray')
        st.write('Use advanced search syntax for precise results:')
        st.markdown('''
            - Surround phrases with quotes (`"`) for exact match.
            - Prepend words or phrases that **must appear** with a `+` symbol. E.g., `+bitcoin`
            - Prepend words that **must not appear** with a `-` symbol. E.g., `-bitcoin`
            - Use `AND`, `OR`, `NOT` keywords, and optionally group with parentheses. E.g., `crypto AND (ethereum OR
            litecoin) NOT bitcoin`
        ''')

        # Input field for the search query
        query = st.text_area('Enter your search query:', placeholder='E.g., crypto AND (ethereum OR litecoin) NOT bitcoin')

        # Section for applying filters
        st.subheader('Filters', divider='gray')

        # Input field for the 'From date' with a default value of today
        from_date = st.date_input(f'From date (limited to {(datetime.today() - relativedelta(months=1)).strftime("%Y-%m-%d")})',
                                  value=datetime.today())

        # Input field for the 'To date' with a default value of today
        to_date = st.date_input('To date', value=datetime.today())

        # Dropdown for selecting the sorting criteria
        sort_by = st.selectbox('Sort by', options=['relevancy', 'popularity', 'publishedAt'])

        # Retrieve the API key from Streamlit secrets
        api_key = st.secrets['news_api_key']

        # Button to trigger the search
        if st.button('Search'):
            # Validate the date inputs
            if from_date > to_date:
                st.error('The "From date" must be earlier than the "To date".')
            elif from_date < (datetime.today().date() - relativedelta(months=1)):
                st.error('The "From date" must be within the last month.')
            elif from_date > datetime.today().date():
                st.error('The "From date" cannot be in the future.')
            elif to_date > datetime.today().date():
                st.error('The "To date" cannot be in the future.')
            elif query:
                # Fetch news articles using the provided inputs
                articles = fetch_news(api_key, query, from_date, to_date, sort_by)

                if articles:
                    # Display the number of articles found
                    st.success(f'Found {len(articles)} articles.')

                    # Loop through and display each article
                    for article in articles:
                        with st.container(border=True):
                            st.subheader(article['title'])  # Article title
                            st.write(f'**{article["source"]["name"]}**')  # Source name
                            st.write(f'Author: *{article["author"]}*')  # Author name
                            if article['urlToImage']:
                                st.image(article['urlToImage'], width=300)  # Article image
                            st.write(article['description'])  # Article description
                            st.write(f'[Read more]({article["url"]})')  # Link to the full article
                            st.write(f'Published on: {article["publishedAt"].split("T")[0]}')  # Publication date
                else:
                    # Display a warning if no articles are found
                    st.warning('No articles found.')
            else:
                # Display an error if the query is empty
                st.error('Please enter a search query.')

    # Add a decorative horizontal line at the bottom of the page
    st.markdown('---')

# Entry point of the script
if __name__ == '__main__':
    main()
