import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops
from towhee.datacollection import DataCollection
import os
from dotenv import load_dotenv

load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI", "https://xxxxxxx:19535")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN",
                         "123123123")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "x3d_m")
MODEL = os.getenv("MODEL", "x3d_m")


def ground_truth(path):
    label = path.split('/')[-2]
    return label_ids[label]


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse video search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 400}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


connections.connect(uri=MILVUS_URI,
                    token=MILVUS_TOKEN)

# collection = create_milvus_collection('x3d_m', 2048)

df = pd.read_csv('./video-data/reverse_video_search.csv')

id_video = df.set_index('id')['path'].to_dict()
label_ids = {}
for label in set(df['label']):
    label_ids[label] = list(df[df['label'] == label].id)


def read_csv(csv_file):
    import csv
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        data = csv.DictReader(f)
        for line in data:
            yield line['id'], line['path'], line['label']


insert_pipe = (
    pipe.input('csv_path')
    .flat_map('csv_path', ('id', 'path', 'label'), read_csv)
    .map('id', 'id', lambda x: int(x))
    .map('path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
    .map('frames', ('labels', 'scores', 'features'),
         ops.action_classification.pytorchvideo(model_name=MODEL, skip_preprocess=True))
    .map(('id', 'features'), 'insert_res',
         ops.ann_insert.milvus_client(uri=MILVUS_URI,
                                      token=MILVUS_TOKEN,
                                      collection_name=DEFAULT_TABLE))
    .output()
)

# insert_pipe('./video-data/reverse_video_search.csv')


# collection.load()

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
