import os
from dotenv import load_dotenv
import pandas as pd
from towhee import pipe, ops
from towhee.datacollection import DataCollection

load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-83c1cf052139d61.api.gcp-us-west1.zillizcloud.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "123")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "x3d_m")
VIDEO_MODEL = os.getenv("MODEL", "x3d_m")


def ground_truth(path):
    label = path.split('/')[-2]
    return label_ids[label]


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
         ops.action_classification.pytorchvideo(model_name=VIDEO_MODEL, skip_preprocess=True))
    .map(('path', 'features'), 'insert_res',
         ops.ann_insert.milvus_client(uri=MILVUS_URI,
                                      token=MILVUS_TOKEN,
                                      collection_name=DEFAULT_TABLE))
    .output()
)

test_pipe = (
    pipe.input('video_file')
    # .flat_map('csv_path', ('id', 'path', 'label'), read_csv)
    # .map('id', 'id', lambda x: int(x))
    .map('video_file', 'frames',
         ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
    .map('frames', ('labels', 'scores', 'features'),
         ops.action_classification.pytorchvideo(model_name=VIDEO_MODEL, skip_preprocess=True))
    .map(('video_file', 'features'), 'insert_res',
         ops.ann_insert.milvus_client(uri=MILVUS_URI,
                                      token=MILVUS_TOKEN,
                                      collection_name=DEFAULT_TABLE))
    .output()
)

# insert_pipe('./video-data/reverse_video_search.csv')
# tt = test_pipe('https://cdn.fanciii-app.com/user/202402290457170000000011/20240229/m86jYaE1.mp4')
# print(tt)


query_pipe = (
    pipe.input('video_file')
    .map('video_file', 'frames',
         ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
    .map('frames', ('labels', 'scores', 'features'),
         ops.action_classification.pytorchvideo(model_name=VIDEO_MODEL, skip_preprocess=True))
    .map('features', 'candidates',
         ops.ann_search.milvus_client(uri=MILVUS_URI,
                                      token=MILVUS_TOKEN,
                                      collection_name=DEFAULT_TABLE, limit=10))
    .output('video_file', 'candidates')
)
res = DataCollection(query_pipe('https://cdn.fanciii-app.com/user/202402290457170000000011/20240229/m86jYaE1.mp4'))

for data in res:
    print(data)
