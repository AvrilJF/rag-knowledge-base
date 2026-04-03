#文档加载 / 解析
import os
import pdfplumber
import docx
import re
import logging
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 抑制pdfplumber警告
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
os.environ["PDFPLUMBER_PARANOID"] = "False"

class DocumentLoader:
    def __init__(self):
        # 文本分块器：按字符数分块，避免上下文过长
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # 每个块500字符
            chunk_overlap=50,     # 块重叠50字符（保证上下文连续）
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
        )

    def _clean_text(self, text):
        """新增：清理乱码、特殊字符、多余空格"""
        # 剔除\u0001这类不可见控制字符
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # 剔除连续空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_file(self, file_path):
        """加载单个文件，返回文本内容"""
        file_ext = os.path.splitext(file_path)[-1].lower()
        text = ""
        
        # PDF解析
        if file_ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += self._clean_text(page_text)  # 调用清理方法
        # Word解析
        elif file_ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += self._clean_text(para.text) + "\n"
        # 纯文本
        elif file_ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = self._clean_text(f.read())
        else:
            raise ValueError(f"不支持的文件格式：{file_ext}")
        
        # 文本分块
        chunks = self.text_splitter.split_text(text)
        # 过滤空分块
        chunks = [chunk for chunk in chunks if chunk.strip()]
        return chunks

# 测试代码
if __name__ == "__main__":
    loader = DocumentLoader()
    chunks = loader.load_file("test.pdf")
    print(f"分块数量：{len(chunks)}")
    print(f"第一个块：{chunks[0][:100]}")