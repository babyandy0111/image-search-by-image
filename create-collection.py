from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from dotenv import load_dotenv

load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-83c1cf052139d61.api.gcp-us-west1.zillizcloud.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "123")
DEFAULT_VIDEO_TABLE = os.getenv("DEFAULT_VIDEO_TABLE", "x3d_m")
DEFAULT_IMAGE_TABLE = os.getenv("DEFAULT_IMAGE_TABLE", "reverse_image_search")
INDEX_TYPE = 'IVF_FLAT'
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "x3d_m")
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DIM = os.getenv("DIM", "2048")


def create_milvus_video_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='seq', dtype=DataType.VARCHAR, descrition='seq', max_length=30, is_primary=True,
                    auto_id=False),
        FieldSchema(name='path', dtype=DataType.VARCHAR, descrition='path to mp4', max_length=500),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='video embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse video search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': METRIC_TYPE,
        'index_type': INDEX_TYPE,
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def create_milvus_image_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='seq', dtype=DataType.VARCHAR, descrition='seq', max_length=30, is_primary=True,
                    auto_id=False),
        FieldSchema(name='path', dtype=DataType.VARCHAR, description='path to image', max_length=500),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': METRIC_TYPE,
        'index_type': INDEX_TYPE,
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    return collection


connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)

create_milvus_video_collection(DEFAULT_VIDEO_TABLE, DIM)
create_milvus_image_collection(DEFAULT_IMAGE_TABLE, DIM)
