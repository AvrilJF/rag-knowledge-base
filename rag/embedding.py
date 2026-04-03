import os
from sentence_transformers import SentenceTransformer

# 设置Hugging Face国内镜像（避免下载卡住）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class TextEmbedding:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        """
        初始化BGE中文向量模型（轻量版，下载快）
        :param model_name: 模型名称，默认轻量版，也可换BAAI/bge-large-zh-v1.5
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text):
        """单文本向量化，返回numpy数组"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
    
    def embed_texts(self, texts):
        """批量文本向量化"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

# 测试代码
if __name__ == "__main__":
    embedder = TextEmbedding()
    text = "RAG知识库问答系统的核心是检索+生成"
    embedding = embedder.embed_text(text)
    print(f"向量维度：{len(embedding)}")
    print(f"向量前5位：{embedding[:5]}")