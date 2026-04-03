#混合检索+重排序
import faiss#Facebook开发的高速向量搜索库
import numpy as np
from rank_bm25 import BM25Okapi
from rag.embedding import TextEmbedding

class HybridRetriever:
    def __init__(self):
        self.embedder = TextEmbedding()
        self.index = None  # FAISS向量索引
        self.texts = []    # 存储所有文本块
        self.bm25 = None   # BM25关键词检索

    def build_index(self, texts):#self是自己这个类的对象、texts是传进来的文档切片（很多段文字）
        """构建向量索引+关键词索引"""
        self.texts = texts# 把传进来的所有文本片段，存到当前对象里

        # 1. 构建FAISS向量索引：按意思、语义检索→ 问 “怎么退货”，能找到 “退款流程”
        embeddings = self.embedder.embed_texts(texts)#把所有文本片段变成向量（数字化）
        dim = len(embeddings[0])# 拿到向量的维度（比如每个向量是1024维，就取1024）
        self.index = faiss.IndexFlatL2(dim)# 创建一个FAISS向量索引（用来做相似度检索）
        # 把文本向量处理成FAISS要求的格式(只接收numpy格式的float32向量)，然后存进向量库
        self.index.add(np.array(embeddings).astype(np.float32)) 
        #np.array转成numpy数组，FAISS只认numpy格式，不认普通列表
        # .astype(np.float32)把数字类型强制变成float32，FAISS要求向量必须是float32格式，否则直接报错！

        #2. 构建关键词检索索引（BM25）：按文字匹配检索→ 问 “怎么退货”，只找带 “退货” 两个字的
        tokenized_texts = [text.split() for text in texts]#列表推导式:texts→text→split()按空格切
        self.bm25 = BM25Okapi(tokenized_texts)# 用分词后的文本创建BM25关键词检索索引
    
    def retrieve(self, query, top_k=5):
        """混合检索：向量检索+关键词检索，合并去重"""
        # 1. 向量检索
        query_embedding = self.embedder.embed_text(query).reshape(1, -1)
        _, vec_indices = self.index.search(query_embedding.astype(np.float32), top_k)
        vec_indices = vec_indices[0].tolist()
        
        # 2. BM25关键词检索
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()
        
        # 3. 合并索引（去重）
        all_indices = list(set(vec_indices + bm25_indices))[:top_k]
        # 4. 返回检索结果
        results = [self.texts[i] for i in all_indices]
        return results

# 测试代码
if __name__ == "__main__":
    retriever = HybridRetriever()
    test_texts = [
        "RAG的全称是检索增强生成（Retrieval-Augmented Generation）",
        "RAG的核心步骤：文档加载→分块→向量化→检索→生成",
        "向量数据库常用的有Milvus、FAISS、Pinecone",
        "混合检索结合了向量检索和关键词检索的优点",
        "重排序可以提升检索结果的准确率"
    ]
    retriever.build_index(test_texts)
    query = "RAG的核心步骤有哪些？"
    results = retriever.retrieve(query)
    print("检索结果：")
    for i, res in enumerate(results):
        print(f"{i+1}. {res}")