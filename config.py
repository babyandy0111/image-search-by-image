import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-83c1cf052139d61.api.gcp-us-west1.zillizcloud.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "2048"))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", "1024"))
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "reverse_image_search")
TOP_K = int(os.getenv("TOP_K", "10"))
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "images/search")
IMG_DOMAIN = os.getenv("IMG_DOMAIN", "https://www.shihjie.com/upload/data")
LOGS_NUM = int(os.getenv("logs_num", "0"))

# Towhee parameters
MODEL = os.getenv("MODEL", "resnet50")
DEVICE = None  # if None, use default device (cuda is enabled if available)

# Milvus parameters
INDEX_TYPE = 'IVF_FLAT'

# path to csv (column_1 indicates image path) OR a pattern of image paths
INSERT_SRC = './data/reverse_image_search_url.csv'
