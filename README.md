# Milvus
Milvus 是一款云原生向量数据库，它具备高可用、高性能、易拓展的特点，用于海量向量数据的实时召回。
##向量数据库
1. 专门设计来存储、索引和查询向量数据的数据库。在人工智能和机器学习领域，数据通常会被转换成高维空间中的向量表示，这样的表示可以捕捉到数据的深层特征和语义信息。向量数据库使得这些高维向量数据的存储和搜索变得高效。
2. 通过计算向量之间的距离，够快速地在大规模数据集中检索到与查询向量最相似的向量。支持图片，视频，分子式，音频
3. 支持高维数据：传统的数据库系统在处理高维数据（如图片、视频、文本转换成的向量）时效率往往不高。向量数据库能够有效管理和查询这类数据。
## 为什么选择Milvus  
1. 支持更多的索引类型和相似度度量方式，提供更多的选择和灵活性
    -索引
        - FLAT：FLAT最适合于在小规模，百万级数据集上寻求完全准确和精确的搜索结果的场景。
        - IVF_FLAT：IVF_FLAT是一种量化索引，最适合于在精度和查询速度之间寻求理想平衡的场景。
        - IVF_SQ8：IVF_SQ8是一种量化索引，最适合于在磁盘、CPU和GPU内存消耗非常有限的场景中显著减少资源消耗。
2. 问答机器人，可集成Cohere, Hugging Face, OpenAI等语言模型制作问答机器人



