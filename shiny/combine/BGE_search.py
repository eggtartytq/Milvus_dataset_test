from pymilvus import connections, utility, MilvusException, FieldSchema, CollectionSchema, DataType, Collection, loading_progress, AnnSearchRequest, RRFRanker
# connections.connect(host="localhost", port="19530", db_name='default')
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

class BGE_search:
    def __init__(self, limit = 5):
       
        self.limit = limit
        self.emb_model = BGEM3EmbeddingFunction(
            model_name='BAAI/bge-m3', # Specify the model name
            device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            )
        self.dense_params = {
            "metric_type" : "L2",
            "params" : {"nprobe":10}  ,
        }
        self.sparse_params ={
            "metric_type" : "IP",
            "params" : {"drop_ratio_build": 0.2},
        }
        self.connections = connections.connect(host="localhost", port="19530", db_name='default')
        self.collection = Collection('Milvus_Test_BGE')

    
    def embed_search(self, input_text, attributes):
        
        search_text = [input_text]
        search_dense = self.emb_model.encode_queries(search_text)['dense']
        search_dense_np = np.array(search_dense)
        search_sparse = self.emb_model.encode_queries(search_text)['sparse']
        search_sparse_np = search_sparse.toarray().ravel()
        output_list = list(attributes)
        
        sparse_result = AnnSearchRequest(
            data = search_sparse,
            anns_field = 'Sparse_embedding',
            param = self.sparse_params,
            limit = self.limit,
            
        )
        
        dense_result = AnnSearchRequest(
            data = search_dense,
            anns_field = 'Dense_embedding',
            param = self.dense_params,
            limit = self.limit,
            
        )
        # import ipdb;ipdb.set_trace()
        output_list.append("Dense_embedding")
        output_list.append("Sparse_embedding")

        result = self.collection.hybrid_search([sparse_result, dense_result], rerank=RRFRanker(), limit=self.limit, output_fields=output_list)
        result_chart = []
        for i in range(self.limit):
            result_frame = {}
            for att in attributes:
                result_frame[att] = getattr(result[0][i].entity, att, None)

            """Dense_cosine_similarity"""
            target_emb = np.array(result[0][i].entity.Dense_embedding)
            # target_emb = result[0][i].entity.Dense_embedding
            dense_similarity = self.__cosine_similarity(search_dense_np, target_emb)
            result_frame['dense_cosine_similarity'] = dense_similarity[0] 
            """Dense_cosine_similarity"""

            """sparse_cosine_similarity"""
            temp_emb = self.__dict_to_csr_matrix(result[0][i].entity.Sparse_embedding, self.emb_model.dim["sparse"])
            sparse_similarity = 1 - cosine(search_sparse_np, temp_emb.toarray().ravel())
            result_frame['sparse_cosine_similarity'] = sparse_similarity
            """sparse_cosine_similarity"""

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
    
    def __dict_to_csr_matrix(self, vector_dict, dimension):
        
        keys = list(vector_dict.keys())
        values = list(vector_dict.values())
        return csr_matrix((values, ([0]*len(keys), keys)), shape=(1, dimension))

if __name__ == "__main__":
    input_string = '10 years'

    obj = BGE_search()
    F = obj.embed_search(input_string, ['Data_Id', 'Label', 'Description'])

    import ipdb;ipdb.set_trace()