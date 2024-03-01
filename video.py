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
         ops.ann_search.milvus_client(uri=MILVUS_URI,
                                      token=MILVUS_TOKEN,
                                      collection_name=DEFAULT_TABLE, limit=10))
    .output('path', 'candidates')
)

res = DataCollection(query_pipe(query_path))

for data in res:
    print(data)
