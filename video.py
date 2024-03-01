from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops
from towhee.datacollection import DataCollection
import os
from dotenv import load_dotenv

load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-83c1cf052139d61.api.gcp-us-west1.zillizcloud.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "123")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "x3d_m")
MODEL = os.getenv("MODEL", "x3d_m")

query_path = './video-data/test/eating_carrots/ty4UQlowp0c.mp4'

query_pipe = (
    pipe.input('path')
    .map('path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
    .map('frames', ('labels', 'scores', 'features'),
         ops.action_classification.pytorchvideo(model_name=MODEL, skip_preprocess=True))
    .map('features', 'result',
         ops.ann_search.milvus_client(uri='https://in01-a0666c0cb6b20dc.aws-us-west-2.vectordb.zillizcloud.com:19535',
                                      token='60f73dd2e01fd8335eab3605911d381f5a9b872be8fe8a170e79b21a435f1b39f05f44fd1e683f85b1393040a988bee5e66f200e',
                                      collection_name=DEFAULT_TABLE, limit=10))
    .map('result', 'candidates', lambda x: [id_video[i[0]] for i in x])
    .output('path', 'candidates')
)

res = DataCollection(query_pipe(query_path))

for data in res:
    print(data)
