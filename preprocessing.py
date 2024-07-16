import pandas as pd
import string


class Preprocess:
    def __init__(self) -> None:
        pass

    def read_csv(
        self,
    ):
        """Read CSV file"""
        self.train_df = pd.read_csv("train.csv", dtype="str")
        # return self.train_df

    def convert_to_str(
        self,
    ):
        """Convert non-string objects in strings for title, author and text"""
        self.train_df["title"] = self.train_df["title"].astype(str)
        self.train_df["author"] = self.train_df["author"].astype(str)
        self.train_df["text"] = self.train_df["text"].astype(str)

    def to_lower(
        self,
    ):
        """Lowercase text data for title, author and text columns"""
        self.train_df["title"] = self.train_df["title"].str.lower()
        self.train_df["author"] = self.train_df["author"].str.lower()
        self.train_df["text"] = self.train_df["text"].str.lower()

    def text_title(
        self,
    ):
        """Create new column called "text_title" merging title and text columns"""
        self.train_df["text_title"] = self.train_df["text"] + self.train_df["title"]

    def remove_punct(
        self,
    ):
        """Remove punctuation, special characters, and extra spaces from text columns"""
        punctuation = (
            string.punctuation + "’" + "'" + '"' + '\r"' + "“" + "'" + '"' + "”" + "”"
        )
        translator = str.maketrans("", "", punctuation)

        # Apply the translation to the specified column and delete any possible double space
        self.train_df["text_title"] = self.train_df["text_title"].str.translate(
            translator
        )
        self.train_df["text_title"] = self.train_df["text_title"].str.replace(
            r"\s+", " ", regex=True
        )
