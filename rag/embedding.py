# -----------------文本转向量的工具（BGE-模型）---------------------
import logging
# 【优化】屏蔽transformers的UNEXPECTED警告，同时避免重复日志
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 导入操作系统模块，用于设置环境变量、管理系统配置
import os

# 导入句子转向量（文本嵌入）核心库
# 作用：把文字转换成计算机能计算的数字向量（embedding）
from sentence_transformers import SentenceTransformer

# -------------------------- 关键配置：加速模型下载 --------------------------
# 设置 Hugging Face 国内镜像地址
# 作用：解决国内网络无法下载模型、下载慢、卡住的问题
# 不设置这行，国内用户几乎一定下载失败
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------- 文本向量化类 --------------------------
class TextEmbedding:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        # 单例：仅第一次创建实例，后续返回同一个实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        """
        初始化BGE中文向量模型（轻量版，下载快）
        :param model_name: 模型名称，默认轻量版，也可换BAAI/bge-large-zh-v1.5
        """
	  # 仅首次初始化加载模型，后续实例化不再重复加载
        if self._model is None:
            print(f"🔄 加载向量模型：{model_name}（仅加载1次）")
	
	
        # 加载预训练好的AI模型到内存
        # BAAI/bge-small-zh-v1.5：百度开源的轻量中文向量模型，速度快、效果好
        self.model = SentenceTransformer(model_name)
        self._model = self.model #ggg
    
    # 方法：给【单个文本】做向量化
    def embed_text(self, text):
        """单文本向量化，返回numpy数组"""
        # 调用模型把文字转向量
        # normalize_embeddings=True：对向量做归一化(长度统一为1 只保留方向)，让检索更准确
        embedding = self.model.encode(text, normalize_embeddings=True)
        # 返回生成好的向量（一维数组，如 512 维）
        return embedding
    
    # 方法：给【多个文本】批量向量化（效率更高）
    def embed_texts(self, texts):
        """批量文本向量化"""
        # 传入文本列表，一次性生成所有向量
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        # 返回向量数组（二维：条数 × 维度）
        return embeddings

# -------------------------- 测试代码 --------------------------
# 只有直接运行这个文件时，才会执行下面代码
if __name__ == "__main__":
    # 创建向量化工具对象（自动下载/加载模型）
    embedder = TextEmbedding()
    
    # 待转换的测试文本
    text = "RAG知识库问答系统的核心是检索+生成"
    
    # 把文本转成向量
    embedding = embedder.embed_text(text)
    
    # 输出向量维度（bge-small-zh-v1.5 固定是 512 维）
    print(f"向量维度：{len(embedding)}")
    
    # 输出向量前5个数字（方便查看）
    print(f"向量前5位：{embedding[:5]}")
