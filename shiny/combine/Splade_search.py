from pymilvus import connections, utility, MilvusException, FieldSchema, CollectionSchema, DataType, Collection, loading_progress
connections.connect(host="localhost", port="19530", db_name='default')
from pymilvus import model
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

class Splade_search:
    def __init__(self, limit = 5):
       
        self.limit = limit
        self.emb_model = model.sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-selfdistil", 
            device="cpu"
        )

        self.index_params = {
            "metric_type" : "IP",
            "params" : {"drop_ratio_search": 0.2}  ,
        }
        self.collection = Collection('Milvus_Test_splade')

    def embed_search(self, input_text, attributes):
        # collection = self.__connect()
        search_text = [input_text]
        search_embedding = self.emb_model.encode_queries(search_text)
        flat_search = search_embedding.toarray().ravel()
        output_list = list(attributes)
        output_list.append('Info_embedding')

        result = self.collection.search(
            data = search_embedding,
            anns_field = 'Info_embedding',
            param = self.index_params,
            limit = self.limit,
            output_fields = output_list
        )

        result_chart = []
        for i in range(self.limit):
            result_frame = {}
            for att in attributes:
                result_frame[att] = getattr(result[0][i].entity, att, None)
            
            """cosine_similarity"""
            temp_emb = self.__dict_to_csr_matrix(result[0][i].entity.Info_embedding, self.emb_model.dim)
            cos_similarity = 1 - cosine(flat_search, temp_emb.toarray().ravel())
            result_frame['cosine_similarity'] = cos_similarity
            """cosine_similarity"""

            result_chart.append(result_frame)
        df_results = pd.DataFrame(result_chart)

        return df_results
    def __dict_to_csr_matrix(self, vector_dict, dimension):
        
        keys = list(vector_dict.keys())
        values = list(vector_dict.values())
        return csr_matrix((values, ([0]*len(keys), keys)), shape=(1, dimension))
    


if __name__ == "__main__":
    input_string = 'education'
    search_rob = Splade_search(limit= 8)
    result = search_rob.embed_search(input_text = input_string, attributes = ['Data_Id', 'Label', 'Description'])

    import ipdb;ipdb.set_trace()