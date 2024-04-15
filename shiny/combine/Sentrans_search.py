from pymilvus import connections, utility, MilvusException, FieldSchema, CollectionSchema, DataType, Collection, loading_progress
# connections.connect(host="localhost", port="19530", db_name='default')
from pymilvus import model
import pandas as pd
import numpy as np

class Sentrans_search:
    def __init__(self, limit = 5):
       
        self.limit = limit
        self.emb_model = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='all-distilroberta-v1', # Specify the model name
            device='cpu' # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )

        self.index_params = {
            "metric_type" : "L2",
            "params" : {"nprobe":10}  ,
        }
        self.connections = connections.connect(host="localhost", port="19530", db_name='default')
        self.collection = Collection('Milvus_Test_Sentrans')

    
    def embed_search(self, input_text, attributes):
        search_text = [input_text]
        search_embedding = self.emb_model.encode_queries(search_text)
        search_emb_np = np.array(search_embedding[0])
        output_list = list(attributes)
        output_list.append('Info_embedding')

        result = self.collection.search(
            data = search_embedding,
            anns_field = 'Info_embedding',
            param = self.index_params,
            limit = self.limit,
            output_fields = output_list
        )
        # import ipdb;ipdb.set_trace()
        result_chart = []
        for i in range(self.limit):
            result_frame = {}
            for att in attributes:
                result_frame[att] = getattr(result[0][i].entity, att, None)

            target_emb = np.array(result[0][i].entity.Info_embedding)
            similarity = self.__cosine_similarity(search_emb_np, target_emb)
            result_frame['cosine_similarity'] = similarity 

            result_chart.append(result_frame)
        

        df_results = pd.DataFrame(result_chart)

        return df_results
    
    def __cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cos_sim = dot_product / (norm_vec1 * norm_vec2)
        # import ipdb;ipdb.set_trace()
        # cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cos_sim


if __name__ == "__main__":
    input_string = '10 years'
    search_rob = Sentrans_search(limit= 8)
    result = search_rob.embed_search(input_text = input_string, attributes = ['Data_Id', 'Label', 'Description'])

    import ipdb;ipdb.set_trace()