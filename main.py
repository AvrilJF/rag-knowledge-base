# 导入操作系统相关功能，比如读取环境变量、创建文件夹
import os

# 导入FastAPI的服务器启动工具，用来运行网页服务
import uvicorn

# 从FastAPI库中导入核心组件：
# FastAPI = 创建网页服务应用
# UploadFile / File = 接收用户上传的文件
# HTTPException = 接口报错时使用
from fastapi import FastAPI, UploadFile, File, HTTPException

# 导入接口返回数据的格式：统一返回JSON格式
from fastapi.responses import JSONResponse

# 导入读取本地.env配置文件的工具（用来存密钥、配置信息）
from dotenv import load_dotenv  # 用于加载本地.env文件

# ===================== 核心兼容逻辑（本地+Docker）=====================
# 1. 加载本地.env文件（本地运行时生效，Docker运行时无影响）
#    - 若不存在.env文件，这行代码不会报错，仅跳过
load_dotenv(override=False)  # override=False：Docker环境变量优先级更高

# 2. 从系统环境变量中读取 阿里云大模型的API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("请配置环境变量 DASHSCOPE_API_KEY：\n"
                     "  本地：在.env文件中配置\n"
                     "  Docker：在docker-compose.yml的environment中配置")

# 把读取到的密钥设置成全局环境变量，让所有代码文件都能使用
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 设置 huggingface 国内镜像地址，没有配置就用默认的，保证模型下载不失败
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")  # 默认值兜底
# ====================================================================

# 导入自己写的RAG功能模块：
# DocumentLoader = 文档加载器，用来读取PDF/Word等文件
from rag.document_loader import DocumentLoader

# HybridRetriever = 混合检索器，用来从知识库找答案
from rag.retriever import HybridRetriever

# AnswerGenerator = 答案生成器，调用大模型生成回答
from rag.generator import AnswerGenerator

# 初始化FastAPI网页服务，设置项目名称、版本、描述
#app = 你的整个Web服务，一个大管家，管理所有接口
app = FastAPI(
    title="RAG企业私有知识库问答系统",
    version="1.0",
    description="兼容本地.env + Docker环境变量的最终版"
)

# 创建全局的检索工具对象（找答案用）
retriever = HybridRetriever()

# 创建全局的答案生成对象（调用AI用），会自动读取环境变量
generator = AnswerGenerator()

# 创建全局的文档加载对象（读文件用）
loader = DocumentLoader()

# 创建一个叫 uploads 的文件夹，用来存用户上传的文件
# exist_ok=True 表示如果文件夹已存在，就不重复创建
os.makedirs("uploads", exist_ok=True)

# ===================== 1. 上传文档接口 =====================
# 接口地址：/upload_doc
# 功能：用户上传文件 → 系统读取 → 存入知识库
@app.post("/upload_doc", summary="上传文档并构建检索索引")
#@装饰器：给下面的函数 “贴标签、注册、绑定”，不强制同名，中间不能有其他代码，否则绑定失败。可以有空行/注释
# 异步函数：接收用户上传的文件，支持PDF/Word/TXT
async def upload_doc(file: UploadFile = File(..., description="支持PDF/Word/TXT格式")):
    #file变量名 :类型标注符号、UploadFile上传文件的类型(FastApi专用)、File来自上传、...必须上传 不传就报错
    try:  # 尝试执行下面的代码，如果出错会跳到except
        # 拼接文件保存路径：uploads/文件名
        file_path = f"uploads/{file.filename}"#f是Python的字符串格式化符号，让字符串里的{变量}能被自动替换成真实的值
        
        # 以二进制写入模式(write binary)打开文件，with:自动帮忙开关文件
        with open(file_path, "wb") as f:#f:给打开的文件起的小名
            f.write(await file.read())#把用户上传的文件内容写入本地
        
        # 切片chunks：把上传的文件读出来→去掉格式/图片/乱码只提取纯文本→按段落+句子+长度→切成一段一段备用
        chunks = loader.load_file(file_path)
        
        # 如果文件内容为空，抛出错误
        if not chunks:
            raise ValueError("文档解析后无有效内容")
        
        # 建立索引：把切好的文本片段，构建成检索索引（知识库）
        retriever.build_index(chunks)
        
        # 返回成功信息：上传成功、分了多少块
        return JSONResponse(
            content={
                "code": 200,
                "msg": "文档上传并构建索引成功",
                "file_name": file.filename,
                "chunk_count": len(chunks)
            }
        )
    # 如果是参数/解析类错误，返回400
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"解析失败：{str(e)}")
    
    # 其他未知错误，返回500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败：{str(e)}")

# ===================== 2. 问答接口 =====================
# 接口地址：/qa
# 功能：用户提问 → 系统检索知识库 → AI生成答案
@app.post("/qa", summary="知识库问答（需先上传文档）")
# 异步函数：接收用户的问题
async def qa(question: str = None, description="用户提问内容"):#None默认值
    # 不写=None表明参数必填，用户不传FastAPI直接拦截报错，不往下跑了
    try:
        # 如果用户没输入问题，直接报错
        if not question:
            raise ValueError("提问内容不能为空")
        
        # 从知识库中检索和问题相关的内容，最多找5条
        context = retriever.retrieve(question, top_k=5)
        
        # 如果没找到相关内容，直接返回未找到
        if not context:
            return JSONResponse(
                content={
                    "code": 200,
                    "question": question,
                    "answer": "未找到相关答案",
                    "context": []
                }
            )
        
        # 调用大模型，根据问题+检索到的内容生成答案
        answer = generator.generate(question, context)
        
        # 返回最终结果：问题、答案、参考资料
        return JSONResponse(
            content={
                "code": 200,
                "question": question,
                "answer": answer,
                "context": context  # 参考文档片段
            }
        )
    
    # 参数错误返回400
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误：{str(e)}")
    
    # 服务错误返回500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{str(e)}")

# ===================== 3. 健康检查接口 =====================
# 接口地址：/health
# 功能：检查服务是否正常运行
@app.get("/health", summary="服务健康检查")
async def health_check():
    return JSONResponse(
        content={
            "code": 200,
            "status": "success",
            "msg": "RAG服务运行正常"
        }
    )

# ===================== 本地运行入口 =====================
# 只有直接运行这个文件时，才会执行下面的代码
if __name__ == "__main__":
    # 打印从环境变量读到的API密钥（调试用）
    # print("DASHSCOPE_API_KEY from env:", os.getenv("DASHSCOPE_API_KEY"))

    # 读取API密钥，只打印前10位+***，保护隐私
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    print("读取到的API Key：", dashscope_api_key[:10] + "***")  # 调试用，验证是否读取成功

    # 尝试初始化AI生成器，校验密钥是否有效
    try:
        generator = AnswerGenerator()#自动读取环境变量
        print("✅ 阿里云百炼大模型初始化成功")
    except ValueError as e:
        print(f"❌ 初始化失败：{str(e)}")
        exit(1)  # 初始化失败，直接退出程序
    
    # 控制台打印启动信息
    print("🚀 启动RAG企业知识库问答系统...")

    # 用uvicorn启动FastAPI服务
    # main:app = 从main.py中加载app应用
    # host=0.0.0.0 = 允许所有设备访问
    # port=8000 = 端口号8000
    # reload=True = 代码修改后自动重启
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式自动重载
        log_level="info"
    )