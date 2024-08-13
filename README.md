# DataLab - Workshop

# Fake News Detection Web App 

## Link --> https://datalab-workshop.streamlit.app/

## Motivation
In the current digital age, the proliferation of fake news has become a significant concern. Misinformation can spread rapidly, misleading readers and causing widespread impact. To address this issue, our team embarked on a project to develop a web application that leverages machine learning techniques to detect fake news.
This tool aims to assist news portals in verifying the authenticity of articles before publication, ensuring that only credible and verified information reaches the audience.


## Goal
The primary goal of this project is to develop an easy-to-use web application for identifying fake news articles. 
At first, it is necessary to create an algorithm that can predict whether a news article is fake or not. The second step consists on the development of the web-app that utilizes this algorithm to accurately identify fake news articles. By providing a reliable verification tool, we aim to enhance the credibility of news portals and contribute to the fight against misinformation.


## Technical Details
Our fake news detection web app is built using Python and incorporates several powerful libraries and tools to achieve high accuracy and performance. Below are the key technical components and libraries used in this project:

### Libraries and Tools:

![images2](https://github.com/user-attachments/assets/b1ec8451-a649-4b09-8d3b-10cf9147f4e3)


* scikit-learn: A robust library for machine learning in Python, used for building and training the fake news detection models.

  ![1_YM2HXc7f4v02pZBEO8h-qw](https://github.com/user-attachments/assets/93aaef9f-6256-48ce-bdc4-f375734ee921)


  
* NLTK (Natural Language Toolkit): A library for working with human language data, used for text processing and feature extraction.
  
![images (1)](https://github.com/user-attachments/assets/be20e503-2bd8-4e3f-9051-592a5ef54865)



* pandas: A data manipulation and analysis library, used for handling datasets.

![NumPy_logo_2020 svg](https://github.com/user-attachments/assets/5442b112-f2f0-41f5-8d21-694069914138)



* numpy: A fundamental package for numerical computing in Python, used for various numerical operations.



#### In addition, some of the tools used for the project:
* UMAP (Uniform Manifold Approximation and Projection): a dimensionality reduction technique that is particularly effective for visualizing high-dimensional data.
* TfidfVectorizer: From the scikit-learn library, used to convert text data into numerical features based on term frequency-inverse document frequency (TF-IDF).
* RandomForestClassifier: An ensemble machine learning algorithm from scikit-learn, used as the primary model for detecting fake news.
  
### Pre-requirements 📋

All the libraries required for using this tool are in the [requirements.txt] file attached.


## Starting 🚀

### How to use this tool
#### For users
If you just want to try this tool, visit https://datalab-workshop.streamlit.app, the URL where the tool is deployed. You can upload your own news file in .csv format to check whether this news is fake or not.

#### For developers
At first, clone this GitHub repo on your local machine and make a pull request if you wish to contribute on it.
To run it from your IDE, please run "streamlit run webapp.py" from your terminal. 
If you are having some trouble, try with "streamlit run webapp.py --server.enableXsrfProtection false"   


## Conclusion
By integrating state-of-the-art machine learning techniques and creating a seamless user experience, our fake news detection web app aims to be a valuable tool for news portals and readers alike, promoting the dissemination of accurate and trustworthy information. 

## Team Members

- **Leonardo Ferreira da Silva** 
  - GitHub: [@leofds12](https://github.com/leofds12)

- **Maria Manuela Saez** 
  - GitHub: [@manuelasaez](https://github.com/manuelasaez)

- **Nataniel Martinez** 
  - GitHub: [@nata3508](https://github.com/Nata3508)

- **Luis Vasquez** 
  - GitHub: [@LuisVas24](https://github.com/LuisVas24)
- **Luciano Darriba** 
  - GitHub: [@lucianodarriba](https://github.com/lucianodarriba)

More information about the project can be found in the complete documentation on Notion: [Fake_news_doc](https://www.notion.so/Fake-News-Project-Documentation-8dfd79c111b04254bc67b1c83e70a940?pvs=4)

## License 📄

This project is under license [LICENSE.md](LICENSE.md) for further details


