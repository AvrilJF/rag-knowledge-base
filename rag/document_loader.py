# -------------------------- 文档加载 / 解析 工具类 --------------------------
import logging
# 屏蔽pdfplumber的FontBBox警告
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# 导入操作系统相关模块，用于处理文件路径、文件后缀
import os

# 导入PDF解析库，专门用来提取PDF里的文本内容：只管 “提取”，不管 “清洗、格式化、分块”
import pdfplumber

# 导入Word文档解析库，用于读取docx格式的Word文件
import docx

# 导入正则表达式库，用于清理文本、过滤特殊字符
import re

# 导入日志模块，用于控制程序日志输出（比如屏蔽警告）
import logging

# 导入LangChain官方推荐的文本分块工具，把长文本切成小段
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------- 配置：屏蔽无关警告 --------------------------
# 只显示 pdfplumber 的 ERROR 级别日志，屏蔽警告、信息类日志
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# 设置环境变量，关闭 pdfplumber 的严格校验，避免多余报错
os.environ["PDFPLUMBER_PARANOID"] = "False"


# -------------------------- 文档加载类：核心功能 --------------------------
class DocumentLoader:
    # 构造函数：创建对象时自动执行，初始化分块器
    def __init__(self):
        # 初始化 文本分块器，用于把长文本切成合适长度的小块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # 每个文本块最大 500 个字符
            chunk_overlap=50,     # 块与块之间重叠 50 字符，保证上下文不中断
            separators=[          # 按这些符号优先分割（优先级从左到右）
                "\n\n", "\n", "。", "！", "？", "，", "、", " "
            ]
        )

    # 内部方法：清理文本乱码、多余空格、不可见字符
    def _clean_text(self, text):
        """新增：清理乱码、特殊字符、多余空格"""
        # 用正则剔除0~31、127号不可见控制字符（如 \u0001空字符），''替换成空字符=删除
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)#re.sub(换什么, 换成啥, 在哪换)
        #[]匹配括号里任意一个字符、\x00-\x1F匹配ASCII码0到31的所有字符、\x7F额外匹配ASCII 127(delete)这一个字符

        # 把连续的空格、换行、制表符替换成单个空格，并去掉首尾空白strip(副本)
        text = re.sub(r'\s+', ' ', text).strip()#\s空格、换行、制表符；+连续出现1次以上。
        
        # 返回清理后的干净文本
        return text

    # 对外方法：接收文件路径，读取文件并返回分块后的文本列表
    def load_file(self, file_path):
        """加载单个文件，返回文本内容"""
        # 获取文件后缀名并转小写（.pdf / .docx / .txt）
        file_ext = os.path.splitext(file_path)[-1].lower()#splitext:根据.拆分文件名a.b.c.pdf拆成(a,.b,.c,.pdf)元组
        
        # 初始化空字符串，用于存储提取的全文
        text = ""
        
        # -------------------------- 1. 处理 PDF 文件 --------------------------
        if file_ext == ".pdf":#file_path[-4:]也能取到
            # 打开PDF文件（with 语法会自动关闭文件，安全）
            with pdfplumber.open(file_path) as pdf:#pdf是一个pdfplumber.PDF实例对象
                # pdfplumber流式逐页读取:必须with,因为它保持文件打开，必须控制生命周期
                # 遍历 PDF 每一页
                for page in pdf.pages:
                    # 提取当前页文本，为空则返回空字符串
                    page_text = page.extract_text() or ""
                    # 清理文本后拼接到总文本
                    text += self._clean_text(page_text)

        # -------------------------- 2. 处理 Word 文件（.docx） --------------------------
        elif file_ext == ".docx":#file_path[-4:]就只能取到docx
            # 加载 Word 文档
            doc = docx.Document(file_path)#docx不用with：因为它一次性读完就关文件，安全省心
            # 遍历 Word 里所有段落
            for para in doc.paragraphs:
                # 清理段落文本并追加，加换行保持格式
                text += self._clean_text(para.text) + "\n"

        # -------------------------- 3. 处理纯文本文件（.txt） --------------------------
        elif file_ext == ".txt":
            # 以 UTF-8 编码打开文本文件
            with open(file_path, "r", encoding="utf-8") as f:
                # open()必须with,凡是Python里支持with的，都是需要手动开关的资源（文件、网络连接、数据库）
                # 读取全部内容并清理
                text = self._clean_text(f.read())

        # -------------------------- 不支持的文件格式 --------------------------
        else:
            # 抛出异常，提示不支持的格式
            raise ValueError(f"不支持的文件格式：{file_ext}")
        
        # -------------------------- 文本分块 --------------------------
        # 把长文本切成预设大小的小块
        chunks = self.text_splitter.split_text(text)
        
        # 过滤掉空的、只有空格的无效块:strip()删掉字符串首尾的空格、换行、制表符,只有这些删完if chunk.strip()=false
        chunks = [chunk for chunk in chunks if chunk.strip()]#列表推导式
        
        # 返回分块后的文本列表
        return chunks


# -------------------------- 测试代码（直接运行此文件时执行） --------------------------
if __name__ == "__main__":
    # 创建文档加载器对象
    loader = DocumentLoader()
    
    # 加载 test.pdf 文件，得到文本块列表
    chunks = loader.load_file("test.pdf")
    
    # 输出分块总数
    print(f"分块数量：{len(chunks)}")
    
    # 输出第一个块的前100个字符
    print(f"第一个块：{chunks[0][:100]}")