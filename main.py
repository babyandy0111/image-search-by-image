import cv2
from towhee.types.image import Image
import csv
from glob import glob
from pathlib import Path

from towhee import pipe, ops, DataCollection
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE, TOP_K, DEFAULT_TABLE, MODEL, DEVICE, \
    INDEX_TYPE, INSERT_SRC


# Load image path
def load_image(x):
    if x.endswith('csv'):
        with open(x) as f:
            reader = csv.reader(f)
            next(reader)
            for item in reader:
                yield item[1]
    else:
        for item in glob(x):
            yield item


def read_images(results):
    images = []
    for key in results:
        path = results[key]
        images.append(Image(cv2.imread(path), 'BGR'))
    return images


# Create milvus collection (delete first if exists)
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='path', dtype=DataType.VARCHAR, description='path to image', max_length=500,
                    is_primary=True, auto_id=False),
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


def my_func(str):
    return str.replace("/Users/andy/PycharmProjects/image-search/", "")


def insert():
    milvus_uri = 'https://in03-83c1cf052139d61.api.gcp-us-west1.zillizcloud.com'
    token = '9d72620ca076ceed08b566c184f92f7e5b1e070ae4ee593b841ce768eee9c353dd82d55243344f7a12c1dcdaaf25c1a41c17ae01'

    p_embed = (
        pipe.input('src')
            .flat_map('src', 'img_path', load_image)
            .map('img_path', 're_img_path', my_func)
            .map('img_path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
    )

    connections.connect(
        # host=MILVUS_HOST,
        # port=MILVUS_PORT
        uri=milvus_uri,
        token=token
    )

    collection = create_milvus_collection(DEFAULT_TABLE, VECTOR_DIMENSION)
    print(f'A new collection created: {DEFAULT_TABLE}')

    p_insert = (
        p_embed.map(('re_img_path', 'vec'), 'mr', ops.ann_insert.milvus_client(
            # host=MILVUS_HOST,
            # port=MILVUS_PORT,
            uri=milvus_uri,
            token=token,
            collection_name=DEFAULT_TABLE
        )).output('mr')
    )

    p_insert(INSERT_SRC)

    # Check collection
    print('Number of data inserted:', collection.num_entities)


def search():
    p_embed = (
        pipe.input('src')
            .flat_map('src', 'img_path', load_image)
            .map('img_path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
    )

    # Search pipeline
    p_search_pre = (
        p_embed.map('vec', ('search_res'), ops.ann_search.milvus_client(
            host=MILVUS_HOST, port=MILVUS_PORT, limit=TOP_K,
            collection_name=DEFAULT_TABLE))
            .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x])
        #                .output('img_path', 'pred')
    )

    p_search = p_search_pre.output('img_path', 'pred')

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=DEFAULT_TABLE)

    # Search for example query image(s)
    collection.load()
    dc = p_search('./data/test/goldfish/*.JPEG')

    # Display search results with image paths
    DataCollection(dc).show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(my_func('/Users/andy/PycharmProjects/image-search/https://www.shihjie.com/upload/data/train/loudspeaker/n03691459_64837.JPEG'))
    insert()
    # search()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
