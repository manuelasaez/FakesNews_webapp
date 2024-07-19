from utils import Preprocess, Vectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)  # Import precision_score, recall_score, and f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from data_modelling import


def main():

#    object_preprocess = Preprocess()
#    object_preprocess.read_csv()
#    object_preprocess.convert_to_str()
#    object_preprocess.remove_rows()
#    object_preprocess.remove_duplicates()
#    object_preprocess.remove_rows_lower_than20()
#    object_preprocess.filter_english_text_edit_df(object_preprocess.train_df, "text")

    preprocess = Preprocess()
    train_df, test_df = preprocess.read_csv()
    train_df, test_df = preprocess.remove_rows()
    train_df, test_df = preprocess.remove_duplicates()
    train_df, test_df = preprocess.remove_rows_lower_than20()
    train_df = preprocess.filter_english_text_edit_df(train_df, 'text')
    test_df = preprocess.filter_english_text_edit_df(test_df, 'text')


    object_vectorization = Vectorization()
    """object_vectorization.create_new_text(object_preprocess.train_df)
    object_vectorization.create_new_text(object_preprocess.test_df)"""
   
    #df_train_en = object_vectorization.filter_english_text_edit_df(
   #     object_preprocess.train_df, "text"
   # )

    labels1 = train_df.copy()["label"].values
    labels = [int(label) for label in labels1]
    filtered_corpus = train_df.copy()["text"].values
 

    max_features = 30  # esto se puede cambiar

    unigram_vectors_without_stopwords = object_vectorization.get_tfidf_vectors(
        filtered_corpus, "english", max_features, 1
    )
   

    X_train, X_test, y_train, y_test = train_test_split(
        unigram_vectors_without_stopwords, labels, random_state=1
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print()
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Performance: LOGISTIC REGRESSION")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print()


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Performance: RANDOM FOREST")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print()

if __name__ == "__main__":
    main()
