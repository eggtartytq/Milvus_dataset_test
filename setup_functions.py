import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# nltk.download('wordnet')
# nltk.download('omw-1.4')  # 为词形还原下载额外的WordNet多语言数据
# nltk.download('punkt')

from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

def Create_Index(collection_Name: str, vector_name: str, index_par: dict):
    try:
        collection = Collection(collection_Name)
        collection.create_index(field_name = vector_name, index_params = index_par)
    except Exception as e:
        print(e)
    else:
        print("Index created")

def nltk_tokenize(text):
    return word_tokenize(text)

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def lemmatize_text(text):
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized_text)

def Insert_vector2(id_list: list, data_file_list: list, collected_list: list, variable_list: list, label_list: list, Description_list: list, embeddings_Description, Collection):
    try:
        collection = Collection
 
        for id in id_list:
            index = id - 1
            entities = [[id_list[index]], [data_file_list[index]], [collected_list[index]], [variable_list[index]], [label_list[index]], [Description_list[index]], [embeddings_Description[index]]]
            collection.insert(entities)
            # print(f"编号{id}插入完毕")
    except Exception as e:
        print(e)