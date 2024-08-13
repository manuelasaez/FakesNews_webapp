import pandas as pd
import numpy as np
import re
import os
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
"""!pip install langdetect
!pip install umap-learn"""
class Preprocess:
    def __init__(self):
        self.df = None

    def read_manual(self, title, author, text):
        self.df = pd.DataFrame([{'id_new': -1,
                                 'title': title,
                                 'author': author,
                                 'text': text}])
        return self.df

    def read_csv(self,  df_file=None):
        """Read CSV files. If test_file is None, only read test file."""
        self.df = pd.read_csv(os.getcwd() + "/data/train.csv", dtype="str")
        #self.test_df = pd.read_csv(os.getcwd() + "/data/test.csv", dtype="str")
        if df_file:
            self.df = pd.read_csv(df_file, dtype="str")
        return self.df

    def remove_rows(self):
        """Remove rows with missing values and convert to string"""
        self.df = self.df.dropna()
        self.df = self.df.astype(str)
        for column in self.df.columns:
            self.df[column] = self.df[column].apply(lambda x: re.sub(r'[^\w\s\d+]', '', x))
        return self.df

    def remove_duplicates(self):
        """Remove duplicate rows"""
        self.df = self.df.drop_duplicates()
        return self.df

    def remove_rows_lower_than20(self):
        """Remove rows where 'text' length is less than 20"""
        self.df = self.df[self.df["text"].str.len() >= 20]
        return self.df


    def newtext(self):
        """Convert non-string objects in strings for title, author and text and then fill missing values with empty strings
        Create new column called "new text" merging title, text and author column"""

        self.df["new_text"] = self.df["text"] + self.df["title"] + self.df["author"]
        
        return self.df

    def filter_english_text_edit_df(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Detects and filters only English text from a DataFrame based on a specified text column.
        Edits the original DataFrame to keep only the rows where the detected language is English.

        Args:
        - df (pd.DataFrame): The original DataFrame containing text data.
        - text_column (str): The name of the column in df containing text data to analyze.

        Returns:
        - pd.DataFrame: Filtered DataFrame containing only rows with English text.
        """
        # Validate that the text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        # Initialize an empty list to store indices of rows to keep
        keep_indices = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            text = row[text_column]
            try:
                # Detect the language of the text
                if detect(text) == "en":
                    # If the language is English, add the index to the list of keep_indices
                    keep_indices.append(index)
            except LangDetectException as e:
                print(f"Language detection failed for row {index}: {text}")
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error for row {index}: {text}")
                print(f"Error: {e}")

        # Filter the original DataFrame to keep only the rows where text is in English
        filtered_df = df.loc[keep_indices].reset_index(drop=True)

        # Check the number of rows before and after filtering
        print(f"Original DataFrame size: {len(df)}")
        print(f"Filtered DataFrame size: {len(filtered_df)}")

        return filtered_df


class Vectorization:
    def __init__(self) -> None:
        pass
    

    def get_tfidf_vectors(
        self,
        corpus: np.ndarray,
        stop_words: str = None,
        max_features: int = None,
        n: int = 1,
    ) -> np.ndarray:
        """
        Vectorizes a corpus of text using TF-IDF (Term Frequency-Inverse Document Frequency).

        Args:
        - corpus (np.ndarray): Array-like, each element is a string representing a document.
        - stop_words (str, optional): Language for stop words ('english', 'spanish', etc.) or None to include all words.
        - max_features (int, optional): Maximum number of features (terms) to consider when vectorizing.
        - n (int, optional): Range of n-grams to consider; (n, n) means only n-grams of size 'n'.

        Returns:
        - np.ndarray: Matrix of TF-IDF vectors where each row corresponds to a document in the corpus.

        Note:
        This function uses sklearn's TfidfVectorizer to compute TF-IDF vectors.
        Each document in the corpus is transformed into a vector representation based on the TF-IDF scores of its terms.
        """

        # Create a TfidfVectorizer object with the given parameters
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words, max_features=max_features, ngram_range=(n, n)
        )

        # Fit the vectorizer to the corpus and transform the text data into TF-IDF vectors
        self.vectorized = self.vectorizer.fit_transform(corpus)
        # Obtener los nombres de las características
        self.words = self.vectorizer
        # Return the resulting TF-IDF vectors
        return self.vectorized, self.words

# Función para dividir autores
def split_authors(author_str):
    """
    Divides a string of authors into a list of individual author names.

    This function takes a string containing author names separated by commas or the word 'and',
    and returns a list of author names. Leading and trailing whitespace is removed from each name.

    Args:
        author_str (str or pd.NaT): A string containing author names separated by ', ' or ' and ',
                                    or a NaN value (pd.NaT). If NaN, an empty list is returned.

    Returns:
        List[str]: A list of author names, with leading and trailing whitespace removed.

    Examples:
        >>> split_authors("John Doe, Jane Smith and Alice Johnson")
        ['John Doe', 'Jane Smith', 'Alice Johnson']
        
        >>> split_authors("Albert Einstein, Marie Curie")
        ['Albert Einstein', 'Marie Curie']
        
        >>> split_authors(None)
        []
    """
    if pd.isna(author_str):
        return []
    authors = re.split(', | and ', author_str)
    return [author.strip() for author in authors]

# Función para verificar si un nombre parece ser de una persona
def is_person_name(lst):
    words = lst.split()
    return len(words) < 4


def authors_counts():
    """
    Reads a CSV file containing news articles and calculates the count of articles for each author.

    This function performs the following tasks:
    1. Reads a CSV file named `train.csv` from the `data` directory in the current working directory.
    2. Calculates the total count of news articles written by each author and adds this information to the DataFrame.
    3. Calculates the count of news articles labeled as '0' (real news) for each author and adds this information to the DataFrame.
    4. Maps these counts to new columns in the DataFrame: 'news_count' and 'real_count'.

    Returns:
        pd.DataFrame: The modified DataFrame with two additional columns:
            - 'news_count': The total number of articles written by each author.
            - 'real_count': The number of articles labeled as '0' (real news) written by each author. If no real news articles exist for an author, the count is set to 0.

    Notes:
        - The CSV file must contain columns named 'author' and 'label'.
        - The function assumes that the current working directory contains a 'data' folder with the `train.csv` file.

    Examples:
        >>> df = authors_counts()
        >>> df.head()
           author  label  news_count  real_count
        0  John Doe      0          10          5
        1  Jane Smith    1          8           3
        2  Alice Johnson 0          10          5
        3  John Doe      1          10          5
        4  Jane Smith    0          8           3
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.getcwd() + "/data/train.csv", dtype="str")
    
    # Calculate the count of news articles for each author
    author_counts = df['author'].value_counts()
    
    # Map the counts back to the DataFrame
    df['news_count'] = df['author'].map(author_counts)
       # Calcula el número de noticias reales (donde 'label' es '0') por autor
    real_news_counts = df[df['label'] == '0']['author'].value_counts()
    
    # Mapea el conteo de noticias reales a la columna 'real_news_count'
    df['real_count'] = df['author'].map(real_news_counts).fillna(0).astype(int)
    
    # Return the modified DataFrame
    return df



# Función para comparar el nuevo DataFrame con el reporte inicial y notificar sobre nuevos autores
def author_parcen_check(report_df, new_authors_df):
    """
    Compares a new DataFrame of authors with an initial report to identify new and existing authors.

    This function performs the following tasks:
    1. Standardizes the author names in both DataFrames by converting them to lowercase and removing duplicates.
    2. Checks each author in the new DataFrame against the list of authors in the initial report.
    3. For each author found in the initial report, retrieves the count of news articles and the count of real news articles.
    4. Generates a results DataFrame with information on whether each author is 'New' or 'Existing', along with their respective news counts and percentage of real news.

    Args:
        report_df (pd.DataFrame): DataFrame containing the initial report with columns 'author', 'news_count', and 'real_count'.
        new_authors_df (pd.DataFrame): DataFrame containing new author information with a column 'author'.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - 'author': The name of the author.
            - 'news_count': The number of news articles written by the author, or 0 if the author is new.
            - 'percentage_real_news': The number of real news articles written by the author, or 0 if the author is new.
            - 'status': 'Existing' if the author was found in the initial report, 'New' otherwise.

    Notes:
        - The function assumes the presence of a column named 'author' in both input DataFrames.
        - The `split_authors` and `is_person_name` functions should be defined elsewhere in the code.
    
    Examples:
        >>> report_df = pd.DataFrame({
        ...     'author': ['John Doe', 'Jane Smith'],
        ...     'news_count': [10, 8],
        ...     'real_count': [5, 3]
        ... })
        >>> new_authors_df = pd.DataFrame({
        ...     'author': ['John Doe', 'Alice Johnson']
        ... })
        >>> author_parcen_check(report_df, new_authors_df)
          author  news_count  percentage_real_news    status
        0  john doe         10                      5  Existing
        1  alice johnson      0                      0      New
    """
    results = []
    report_df['author'] = report_df['author'].str.lower().dropna().drop_duplicates()
    new_authors_df['author'] = new_authors_df['author'].str.lower().dropna().drop_duplicates()
    initial_authors = report_df['author'].tolist()

    for _, row in new_authors_df.iterrows():
        authors = split_authors(row['author'])
        for author in authors:
            if is_person_name(author):
                if author in initial_authors:
                    author_report = report_df[report_df['author'] == author]
                    news_count = author_report['news_count'].values[0]
                    percentage_real_news = author_report['real_count'].values[0]
                    results.append({'author': author, 'news_count': news_count, 'percentage_real_news': percentage_real_news, 'status': 'Existing'})
                else:
                    results.append({'author': author, 'news_count': 0, 'percentage_real_news': 0, 'status': 'New'})

    results_df = pd.DataFrame(results)
    return results_df
