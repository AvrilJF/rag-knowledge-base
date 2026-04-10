# -------------------------- 混合检索+重排序 --------------------------
import faiss#Facebook开发的高速向量搜索库

# 导入 numpy 库，简称 np
# 作用：专门用来做 数值计算、向量运算、矩阵处理（AI/向量检索必用库）
import numpy as np
import json
import os
# 从 rank_bm25 库中导入 BM25Okapi 类
# BM25Okapi = 经典的关键词检索算法（搜索引擎底层算法）
# 作用：根据关键词匹配文本，计算相关性分数
from rank_bm25 import BM25Okapi

from rag.embedding import TextEmbedding

import logging

# 屏蔽pdfplumber的FontBBox警告
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
class HybridRetriever:
    _instance = None  # 单例模式，确保全局唯一实例
    _index_path = "data/faiss.index"
    _texts_path = "data/texts.json"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self):
        # 确保仅初始化一次
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.embedder = TextEmbedding()
        self.embeddings = None
        self.index = None  # FAISS向量索引
        self.texts = []    # 存储所有文本块
        self.bm25 = None   # BM25关键词检索
	# 创建数据存储目录
        os.makedirs("data", exist_ok=True)
        # 启动时加载已有索引（如果存在）
        self._load_index()


    def _load_index(self):
        """从本地加载索引，重启服务无需重新上传"""
        if os.path.exists(self._index_path) and os.path.exists(self._texts_path):
            try:
                # 加载FAISS索引
                self.index = faiss.read_index(self._index_path)
                # 加载文本列表
                with open(self._texts_path, "r", encoding="utf-8") as f:
                    self.texts = json.load(f)
                # 重新初始化BM25
                tokenized_texts = [text.split() for text in self.texts]
                self.bm25 = BM25Okapi(tokenized_texts)
                print(f"✅ 从本地加载索引成功，共{len(self.texts)}段文本")
            except Exception as e:
                print(f"⚠️ 加载本地索引失败，将重新构建：{str(e)}")
                self.index = None
                self.texts = None
                self.bm25 = None

    def _save_index(self):
        """将索引保存到本地，持久化存储"""
        if self.index is not None and self.texts is not None:
            # 保存FAISS索引
            faiss.write_index(self.index, self._index_path)
            # 保存文本列表
            with open(self._texts_path, "w", encoding="utf-8") as f:
                json.dump(self.texts, f, ensure_ascii=False, indent=2)
            print(f"✅ 索引已保存到本地，共{len(self.texts)}段文本")

    def build_index(self, texts):#self是自己这个类的对象、texts是传进来的文档切片（很多段文字）
        """构建向量索引+关键词索引"""
        self.texts = texts# 把传进来的所有文本片段，存到当前对象里

        # 1. 构建FAISS向量索引：按意思、语义检索→ 问 “怎么退货”，能找到 “退款流程”
        # embeddings = self.embedder.embed_texts(texts)#把所有文本片段变成向量（数字化）
        # dim = len(embeddings[0])# 拿到向量的维度（比如每个向量是1024维，就取1024）
        # self.index = faiss.IndexFlatL2(dim)# 创建一个FAISS向量索引（用来做相似度检索）
        # 把文本向量处理成FAISS要求的格式(只接收numpy格式的float32向量)，然后存进向量库
        # self.index.add(np.array(embeddings).astype(np.float32)) 
        #np.array转成numpy数组，FAISS只认numpy格式，不认普通列表
        # .astype(np.float32)把数字类型强制变成float32，FAISS要求向量必须是float32格式，否则直接报错！
        self.embeddings = self.embedder.embed_texts(texts)
        # 初始化FAISS索引
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))

        #2. 构建关键词检索索引（BM25）：按文字匹配检索→ 问 “怎么退货”，只找带 “退货” 两个字的
        # tokenized_texts = [text.split() for text in texts]#列表推导式:texts→text→split()按空格切
        # self.bm25 = BM25Okapi(tokenized_texts)# 用分词后的文本创建BM25关键词检索索引
        tokenized_texts = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        # 保存到本地
        self._save_index()

    def retrieve(self, query, top_k=5):
        """混合检索：向量检索+关键词检索，合并去重"""
        # 防御性检查：确保索引已构建
        if self.index is None or self.bm25 is None or self.texts is None:
            raise ValueError("知识库索引未构建，请先上传文档")
        
        # ---------------------- 1. 第一步：向量检索（语义匹配） ----------------------
        # 把问题转为向量，并把形状变成[1, 维度]，适配FAISS检索格式
        query_embedding = self.embedder.embed_text(query).reshape(1, -1)
        # FAISS向量库搜索：
        # 输入：问题向量、要找top_k条
        # 输出：相似度得分(_)、命中的文本索引编号(vec_indices)
        _, vec_indices = self.index.search(query_embedding.astype(np.float32), top_k)
        # 把检索结果从numpy数组转成普通Python列表，方便后续处理
        vec_indices = vec_indices[0].tolist()

        # ---------------------- 2. 第二步：BM25关键词检索（精确匹配） ----------------------
        # 对问题按空格切词
        tokenized_query = query.split()
        # 用 BM25 计算【问题中的词】和【所有文本】的相关度分数
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 对分数从高到低排序，取前 top_k 个，得到关键词检索的索引列表
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        # ---------------------- 3. 第三步：合并结果 + 去重 ----------------------
        # 把向量检索结果 + 关键词检索结果合并，用 set 去重（避免重复返回同一段）
        # 最后再截取 top_k 条，保证结果数量不超标
        all_indices = list(set(vec_indices + bm25_indices))[:top_k]
         # 按向量相似度排序，保证结果相关性
        all_indices_sorted = sorted(all_indices, key=lambda x: -self.embeddings[x] @ query_embedding[0])[:top_k]
        
        # 4. 返回检索结果
        results = [self.texts[i] for i in all_indices_sorted]
        # # ---------------------- 4. 第四步：根据索引取出真实文本 ----------------------
        # # 遍历去重后的索引编号，从原始文本列表中取出对应的文本内容
        # results = [self.texts[i] for i in all_indices]
        
        # 返回最终检索到的高相关文本
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
