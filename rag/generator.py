# 导入操作系统模块，用于读取环境变量（API Key）
import os

# 导入HTTP状态码工具（如200=成功，404=找不到），用于校验大模型返回状态
from http import HTTPStatus

# 导入阿里云百炼 SDK（调用通义千问大模型的官方库）
import dashscope

# 从百炼SDK导入文本生成接口（专门用来对话/问答）
from dashscope import Generation

# 导入LangChain的提示词模板工具，用于拼接系统提示词 + 问题 + 参考资料
from langchain_core.prompts import PromptTemplate

# -------------------------- 全局配置 --------------------------
# 设置Hugging Face国内镜像加速（虽然这里主要用百炼，但保持统一配置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------- 答案生成类 --------------------------
class AnswerGenerator:
    # 构造函数：创建对象时自动执行，初始化大模型配置
    def __init__(self):
        """初始化阿里云百炼大模型生成器"""
        # 1. 从【系统环境变量】中读取 阿里云百炼 API Key
        # 作用：安全的存储密钥，不写死在代码里
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        # 如果没读到API Key，直接抛出错误，防止程序运行失败
        if not self.dashscope_api_key:
            raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置，请检查 docker-compose.yml 或系统环境变量")
        # 把API Key配置到百炼SDK中，完成鉴权（告诉阿里云你是谁）
        dashscope.api_key = self.dashscope_api_key
        # 2. 定义提示词模板（给AI的规则手册）
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
            # 模板里的两个可变参数：上下文资料、用户问题
            input_variables=["context", "question"]
        )

    # 核心方法：接收【用户问题】+【检索到的资料】，返回AI生成的答案
    def generate(self, question, context):
        """生成答案：（适配message格式+清理后上下文）"""
        # 1. 把检索到的多条文本，拼接成带编号的格式：enumerate遍历列表时，同时拿到「索引序号」和「元素内容」
        # 例如：【1】xxx 【2】xxx，方便模型识别和引用
        context_str = "\n".join([f"【{i+1}】{text}" for i, text in enumerate(context)])
        # 2. 拼接最终给大模型的完整提示词
        prompt = f"""请根据以下参考文档，简洁准确地回答用户问题，仅用文档中的信息，不要编造。
        参考文档：
        {context_str}
        用户问题：{question}
        回答："""
        # 异常捕获：防止网络错误/模型报错导致程序崩溃
        try:
            # 调用阿里云百炼大模型（核心代码）
            response = Generation.call(
                model="qwen-turbo",            # 使用的模型：千问turbo（快、便宜）
                messages=[{"role": "user", "content": prompt}],  # 对话格式（必须是list+dict）
                result_format="message",        # 官方推荐格式，兼容性最好
                temperature=0.1,                # 温度越低，回答越严谨、不发散
                top_p=0.8,                      # 控制随机性，配合temperature使用
                max_tokens=1024,                # 最大生成字数（防止太长）
                retry=3                        # 失败自动重试3次
            )
            # -------------------------- 严格校验模型返回结果 --------------------------
            # 如果HTTP状态不是200（成功），返回错误信息
            if response.status_code != HTTPStatus.OK:
                return f"大模型调用失败：{response.code} - {response.message}"
            # 如果模型没有返回任何结果
            if not response.output or not response.output.choices:
                return "未找到相关答案（大模型未返回结果）"
            # 如果返回结果为空内容
            if not response.output.choices[0].message or not response.output.choices[0].message.content:
                return "未找到相关答案（文档中无对应信息）"
            # 一切正常，返回清理后的答案（去掉首尾空格换行）
            return response.output.choices[0].message.content.strip()  
        # 捕获所有未知异常
        except Exception as e:
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈，方便调试
            return f"大模型调用异常：{str(e)}"

# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 测试时手动设置API Key（正式环境用环境变量）
    c = "替换成你的阿里云百炼API Key"
    
    # 创建答案生成器
    generator = AnswerGenerator()
    
    # 测试问题
    question = "RAG的核心步骤有哪些？"
    
    # 模拟检索出来的参考资料
    context = [
        "RAG的全称是检索增强生成（Retrieval-Augmented Generation）",
        "RAG的核心步骤：文档加载→分块→向量化→检索→生成",
        "向量数据库常用的有Milvus、FAISS、Pinecone"
    ]
    
    # 调用生成方法
    answer = generator.generate(question, context)
    
    # 打印结果
    print("生成答案：")
    print(answer)