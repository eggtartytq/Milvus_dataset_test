import time
import pandas as pd
from pymilvus import connections, utility, MilvusException, FieldSchema, CollectionSchema, DataType, Collection, loading_progress
from pymilvus import model

connections.connect(host="localhost", port="19530", db_name='default')
from setup_functions import *

COLLECTION_ID='Milvus_Test_splade'
file_path = './data_WHI.csv'




#set up Schema
fields = [
    FieldSchema(name="Data_Id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="Dataset_File", dtype=DataType.VARCHAR,max_length=255),
    FieldSchema(name="Collected", dtype=DataType.VARCHAR,max_length=100),
    FieldSchema(name="Variable", dtype=DataType.VARCHAR,max_length=100),
    FieldSchema(name="Label", dtype=DataType.VARCHAR,max_length=255),
    FieldSchema(name="Description", dtype=DataType.VARCHAR,max_length=1024),
    FieldSchema(name="Info_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
]

collection_schema = CollectionSchema(fields, description="Example_collection")

doc_data = Collection(COLLECTION_ID, collection_schema)

#index params
index_params = {
    
    "index_type" : "SPARSE_INVERTED_INDEX",
    "metric_type" : "IP",
    "params" : {"drop_ratio_build": 0.2}  ,
}

Create_Index(COLLECTION_ID, "Info_embedding", index_params)
#load index
Collection_Name = COLLECTION_ID
collection = Collection(Collection_Name)
collection.load()

splade_ef = model.sparse.SpladeEmbeddingFunction(
    model_name="naver/splade-cocondenser-selfdistil", 
    device="cpu"
)



df = pd.read_csv(file_path)

id_list = df.index.tolist()
id_plus_one = [x + 1 for x in id_list]

dataset_file_data = df["Dataset File"].tolist()
collected_data = df.Collected.tolist()
variable_data = df.Variable.tolist()



df['Label'] = df['Label'].fillna(' ')
label_data = df['Label'].tolist()

df['Description'] = df['Description'].fillna(' ')
description_data = df['Description'].tolist()

df['combine'] = df['Label'] + df['Description']
df['combine'] = df['combine'].str.replace('[^a-zA-Z\s]', '', regex=True)
df['combine'] = df['combine'].str.lower()
df['combine_no_stopwords'] = df['combine'].apply(remove_stopwords)
df['lemmatize_tokens'] = df['combine_no_stopwords'].apply(lemmatize_text)
combine_data = df['lemmatize_tokens'].tolist()
# import ipdb;ipdb.set_trace()

#embedding
Info_embedding = splade_ef.encode_documents(combine_data)


# import ipdb;ipdb.set_trace()


start_time = time.time()
try:
    entities = [id_plus_one, dataset_file_data, collected_data, variable_data, label_data, description_data, Info_embedding]
    collection.insert(entities)
    collection.flush()
except Exception as e:
    print(f"插入数据时发生错误：{e}")
end_time = time.time()
print(f"插入数据用时为: {end_time - start_time}秒, 平均一条的插入时间为: {(end_time - start_time)/len(id_list)}")
print(f"总共有: {len(id_list)}条数据")