import os
from http import HTTPStatus
import dashscope
from dashscope import Generation
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

# 设置Hugging Face国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class AnswerGenerator:
    def __init__(self):
        """初始化阿里云百炼大模型生成器（最终版）"""
        # 1. 获取并校验百炼API Key
         # 从环境变量读取 DASHSCOPE_API_KEY
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.dashscope_api_key:
            raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置，请检查 docker-compose.yml 或系统环境变量")
        
        # 配置 dashscope
        dashscope.api_key = self.dashscope_api_key
        
        # 2. 提示词模板（优化版，更贴合企业场景）
        self.prompt_template = PromptTemplate(
            template="""
            你是专业的企业知识库问答助手，严格按照以下规则回答：
            1. 仅使用提供的参考文档内容回答，绝不编造任何信息；
            2. 如果参考文档中无相关信息，直接回复“未找到相关答案”；
            3. 回答简洁、准确，分点说明（如有必要），并标注引用来源编号；
            4. 禁止泄露参考文档外的任何信息。
            
            参考文档：
            {context}
            
            用户问题：{question}
            
            回答：
            """,
            input_variables=["context", "question"]
        )
    
def generate(self, question, context):
    """生成答案：最终稳定版（适配message格式+清理后上下文）"""
    # 1. 拼接清理后的参考文档
    context_str = "\n".join([f"【{i+1}】{text}" for i, text in enumerate(context)])
    # 2. 简化提示词，明确指令（避免模型困惑）
    prompt = f"""请根据以下参考文档，简洁准确地回答用户问题，仅用文档中的信息，不要编造。
        参考文档：
        {context_str}

        用户问题：{question}
        回答："""    
    try:
        # 核心修改：改用result_format="message"（百炼官方推荐）
        response = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",  # 替换为message模式
            temperature=0.1,
            top_p=0.8,
            max_tokens=1024,
            retry=3
        )
        
        # 严格的空值校验
        if response.status_code != HTTPStatus.OK:
            return f"大模型调用失败：{response.code} - {response.message}"
        if not response.output or not response.output.choices:
            return "未找到相关答案（大模型未返回结果）"
        if not response.output.choices[0].message or not response.output.choices[0].message.content:
            return "未找到相关答案（文档中无对应信息）"
        
        return response.output.choices[0].message.content.strip()  
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"大模型调用异常：{str(e)}"
    
# 测试代码（验证百炼API是否可用）
if __name__ == "__main__":
    # 设置百炼API Key（测试用，实际部署用环境变量）
    os.environ["DASHSCOPE_API_KEY"] = "替换成你的阿里云百炼API Key"
    
    generator = AnswerGenerator()
    question = "RAG的核心步骤有哪些？"
    context = [
        "RAG的全称是检索增强生成（Retrieval-Augmented Generation）",
        "RAG的核心步骤：文档加载→分块→向量化→检索→生成",
        "向量数据库常用的有Milvus、FAISS、Pinecone"
    ]
    
    answer = generator.generate(question, context)
    print("生成答案：")
    print(answer)