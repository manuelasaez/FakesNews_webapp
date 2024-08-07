import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from utils_4 import Preprocess, Vectorization, author_parcen_check, authors_counts
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)  # Import precision_score, recall_score, and f1_score
# Función para cargar el modelo
# Función para cargar el modelo
@st.cache_resource
def load_model(file_name):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    # Título de la aplicación
    st.title("Fake News Text Classification")
    Upload=True #identificador para cargar datos 
    uploaded_file= False #identificador para cargar datos 
    authors = authors_counts() #carga los autores
    # Selector de modelos  
    st.sidebar.markdown('')
    model_option = st.sidebar.selectbox(
        'Select a model to evaluate:',
        ('Random Forest', 'Logistic Regression', 'Gaussian NB')
    )

    # Cargar el modelo seleccionado
    model_file_map = {
        'Random Forest': 'random_forest_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Gaussian NB': 'gaussian_NB_model.pkl'
    }
    model = load_model(model_file_map[model_option])
    st.write(f"Selected Model: {model_option}")
    
    option = st.sidebar.selectbox(
        'Select an option:',
        ('','View Model Performance', 'Analyze Training Data', 'View Training Data Statistics')
    )    
    # Selector de modelos
    if option == 'View Model Performance':
        #st.write(f"Modelo seleccionado: {model_option}")
        accuracy = model['accuracy']
        precision = model['precision']
        recall = model['recall']
        f1 = model['f1']
        st.write(f"Performance:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")
    
    # Selector de modelos
    #w_vecgor = model['vector']
    if option == 'View Training Data Statistics':
        #st.write(f"Modelo seleccionado: {model_option}")
        w_vecgor = model['vector']
        feature_names = w_vecgor.get_feature_names_out()
        st.write("The 100 words selected by the TF-IDF model are:")
        #st.write(feature_names)
        
        X_train=model['X_train']
        Y_train=model['Y_train']
        # Convertir X_train a denso si el modelo es Gaussian NB
        #if model_option == 'Gaussian NB':
        #    X_train = X_train.toarray()
        # Filtrar las noticias verdaderas y falsas
        true_indices = [i for i, label in enumerate(Y_train) if label == 0]
        false_indices = [i for i, label in enumerate(Y_train) if label == 1]
        
        true_matrix = X_train[true_indices]
        false_matrix = X_train[false_indices]
        
        # Sumar las frecuencias de las palabras para las noticias verdaderas y falsas
        if  model_option == 'Gaussian NB':
            true_word_counts = true_matrix.sum(axis=0)
            false_word_counts = false_matrix.sum(axis=0)
        else:
            true_word_counts = true_matrix.sum(axis=0).A1
            false_word_counts = false_matrix.sum(axis=0).A1
        
        true_word_freq = dict(zip(feature_names, true_word_counts))
        false_word_freq = dict(zip(feature_names, false_word_counts))
        
        # Crear nubes de palabras
        wordcloud_true = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(true_word_freq)
        wordcloud_false = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(false_word_freq)
        
        st.write("Word Cloud for True News")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_true, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        st.write("Word Cloud for False News")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_false, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    if option == 'Analyze Training Data':
        #st.write(f"Confusion matrix: {model_option}")
        # Graficar la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(model['confusion_matrix'], annot=True, fmt='d', cmap='Blues',cbar=False)
        plt.title(f'Confusion Matrix for {model_option}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

# Función principal de la aplicación Streamlit
    
    # Inicializar variables en st.session_state
    if 'author' not in st.session_state:
        st.session_state['author'] = "Name"
    if 'title' not in st.session_state:
        st.session_state['title'] = "more than three words"
    if 'text' not in st.session_state:
        st.session_state['text'] = "more than twenty words"
  
    # Opción de introducir datos manualmente
    st.markdown('<p style="color: blue;font-weight: bold;">OPTION 1: Manual Data Entry</p>', unsafe_allow_html=True)

    # Crear campos de entrada para el usuario
    author = st.text_input("Author", "")
    title = st.text_input("Title", "")
    text = st.text_area("Text", "")

    # Botón para introducir datos manualmente
    if st.button("Submit"):
        st.write("You have submitted the following data:")
        st.write(f"Author: {author}")
        st.write(f"Title: {title}")
        st.write(f"Text: {text}")

    # Opción de cargar un archivo CSV
    else:
        st.markdown('<p style="color: blue; font-weight: bold; ">OPTION 2: Upload a file</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        preprocess = Preprocess()
        test_df = preprocess.read_csv(uploaded_file)
        st.dataframe(test_df)

        # Verificar si existen las columnas 'title', 'author' y 'text'
        required_columns = ['title', 'author', 'text']
        missing_columns = [col for col in required_columns if col not in test_df.columns]

        if missing_columns:
            st.warning(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}. Please add them and try again.")

        Upload = False  # identificador para cargar datos
        result = author_parcen_check(authors, test_df)
        st.write("File uploaded:", uploaded_file.name)
        st.dataframe(result)
        
    if Upload:
        preprocess = Preprocess()
        test_df = preprocess.read_manual(title,author,text)
        result = author_parcen_check(authors, test_df)
        st.dataframe(result)
        #test_df = pd.DataFrame([{'title': st.session_state['title'],
        #                         'author': st.session_state['author'],
        #                         'text': st.session_state['text']}])
        

    if st.button("Evaluate", key="evaluate_button"):
        if 'author' in test_df.columns and 'title' in test_df.columns and 'text' in test_df.columns:
            # Procesar datos
             
            test_df = preprocess.remove_rows()
            test_df = preprocess.remove_duplicates()
            test_df = preprocess.remove_rows_lower_than20()
            test_df = preprocess.newtext()
            test_df = preprocess.filter_english_text_edit_df(test_df, 'new_text')

            # Vectorizar el texto
            object_vectorization = Vectorization()
            filtered_corpus = test_df["new_text"].values
            max_features = 100  # esto se puede cambiar
            unigram_vectors_without_stopwords, _ = object_vectorization.get_tfidf_vectors(filtered_corpus, "english", max_features, 1)

            if unigram_vectors_without_stopwords.shape[1] < 100:
                st.warning(f"The document does not have enough unique words. It has {unigram_vectors_without_stopwords.shape[1]} out of the 100 it should have.")
            else:
                # Hacer predicciones
                model_vec=model['model']
                predictions = model_vec.predict(unigram_vectors_without_stopwords)
    
                # Mostrar las predicciones
                test_df['label_predict'] = predictions
                
                st.dataframe(test_df)
                
    
                def highlight_predictions(row):
                    color = 'background-color: red' if row['label_predict'] == 1 else ''
                    return [color] * len(row)
    
                styled_df = test_df.style.apply(highlight_predictions, axis=1)
                st.write("The ones marked in red are Fake News")
                st.dataframe(styled_df)

        else:
            st.warning("Please fill in all fields.")

    # Guardar nuevos datos si no hay duplicados
    if st.button("Save Data"):
        save_path = 'data/saved_data.csv'

        if os.path.exists(save_path):
            existing_data = load_data(save_path)
            st.write(test_df)            
            st.dataframe(existing_data)
        
        else:
            existing_data = pd.DataFrame(columns=['id_new', 'title', 'author', 'text', 'label_predict'])
            st.dataframe(existing_data)
        
        
        duplicates = existing_data.merge(test_df, on=['text'], how='inner')

        if not duplicates.empty:
            st.warning("Duplicate entries found. Please confirm to overwrite.")
            st.dataframe(duplicates)
            st.session_state['duplicate_entries'] = duplicates
            if st.button("Confirm Save Duplicates"):
                if not st.session_state['duplicate_entries'].empty:
                    combined_data = pd.concat([existing_data, test_df]).drop_duplicates().reset_index(drop=True)
                    st.write("Author introduced:", len(styled_df))
                    combined_data['id_new'] = range(0,len(combined_data))
                    combined_data.to_csv(save_path, index=False)
                    st.success("Duplicate data saved successfully!")
                    st.session_state['duplicate_entries'] = pd.DataFrame()  # Clear duplicate entries after saving
                else:
                    st.warning("No duplicates to save.")
        else:
            combined_data = pd.concat([existing_data, test_df]).drop_duplicates().reset_index(drop=True)
            combined_data['id_new'] = range(0,len(combined_data))
            combined_data.to_csv(save_path, index=False)
            st.write("File Saved")
            st.dataframe(duplicates)
            st.success("Data saved successfully!")

if __name__ == "__main__":
    main()

