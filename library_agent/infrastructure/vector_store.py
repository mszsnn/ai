# 处理向量数据库
#---------------------
# 1 创建持久化链接
# 2 不同的书本进行物理隔离
#---------------------
import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "rag_engine" / "db_vector_data"

load_dotenv()

logger = logging.getLogger(__name__)

class MultiTenantVectorStore:

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        # 创建文件夹
        os.makedirs(db_path, exist_ok = True)
        # 创建持久化链接
        self.client = chromadb.PersistentClient(
            path = db_path
        )
        # 翻译矩阵模型
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="text-embedding-3-small"
        )

        logger.info(
            "vector_store_ready",
            extra={"event": "vector_store_ready", "db_path": str(db_path)},
        )



    def create_collection_by_tenant_id(self, tenant_id : str= ''):
        """
        删除原有向量空间，创建新的向量空间
        """
        # 这里有个细节我们， 很多情况下我们需要重新创建某个 id 的向量空间
        try:
            self.client.delete_collection(name=tenant_id)
            logger.info(
                "vector_collection_reset",
                extra={"event": "vector_collection_reset", "book_id": tenant_id},
            )
        except (ValueError, chromadb.errors.NotFoundError):
            pass

        # 根据 tenant_id 获取独立逻辑向量空间， 如果没有就创建，有返回
        collection = self.client.create_collection(
            name=tenant_id,
            embedding_function=self.embedding_fn,
            metadata={ 'description': f'Knowledge for {tenant_id}'}
        )
        return collection

    def get_collection_by_tenant_id(self, tenant_id : str= ''):
        """
        检索专用
        """
        return self.client.get_collection(
            name=tenant_id,
            embedding_function=self.embedding_fn
        )



# 全局单例， 就不用传递
_global_vector_store = None

def get_global_vector_store( db_path: str = DEFAULT_DB_PATH):
    """
    全局唯一链接，方便使用
    """
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = MultiTenantVectorStore(db_path=db_path)
    return _global_vector_store
