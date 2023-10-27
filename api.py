import base64
import os
from urllib.parse import urlparse
import cv2
from towhee import pipe, ops, DataCollection
from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv

load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-83c1cf052139d61.api.gcp-us-west1.zillizcloud.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "123123123123")
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


def my_func(path_list):
    return [str(y[0]) for y in path_list]


def makedir(dir_path):
    dir_path = os.path.dirname(dir_path)
    bool = os.path.exists(dir_path)
    if bool:
        pass
    else:
        os.makedirs(dir_path)


def get_unique_list(data):
    unique = []

    for path in data:
        if path in unique:
            continue
        else:
            unique.append(path)
    return unique


p2 = (
    pipe.input('url')
        .map('url', 'original', ops.image_decode())
        .map('original', ('box', 'class', 'score'), ops.object_detection.yolov5())
        .map(('original', 'box'), 'object', ops.towhee.image_crop(clamp=True))
        .map('original', 'original_embedding', ops.image_embedding.timm(model_name=MODEL))
        .map('object', 'object_embedding', ops.image_embedding.timm(model_name=MODEL))
        .output('url', 'original', 'box', 'object', 'class', 'object_embedding', 'original_embedding', 'score')
)

p_search = (
    pipe.input('img_embedding')
        .map('img_embedding', ('search_res'),
             ops.ann_search.milvus_client(uri=MILVUS_URI, anns_field='embedding',
                                          token=MILVUS_TOKEN, limit=TOP_K,
                                          collection_name=DEFAULT_TABLE))
        .map('search_res', 'pred', my_func)
        .output('pred')
)

app = FastAPI()


@app.get('/search')
async def search_images(image: str = ''):
    try:
        if image == '':
            return {'status': False, 'msg': "image is None"}, 400

        url = urlparse(image)
        fileName = os.path.basename(url.path)
        content = image
        res = p2(content)
        jsonObj = {}
        outClass = []
        outBox = []
        outObject = []
        outOriginal = []
        outObjectResult = []
        outOriginalResult = []
        objectEmbedding = []
        originalEmbedding = []
        for data in DataCollection(res):
            _, buffer = cv2.imencode('.jpg', data['original'])
            originalBase64 = base64.b64encode(buffer.tobytes())
            outOriginal.append(originalBase64)

            for r in data['class']:
                outClass.append(r)

            for r in data['box']:
                outBox.append(r)

            # 將圖片轉成base64
            i = 0
            for r in data['object']:
                objectPath = UPLOAD_PATH + '/' + data['class'][i] + '/' + data['class'][i] + '-object-' + fileName
                makedir(objectPath)
                cv2.imwrite(objectPath, r)
                _, buffer = cv2.imencode('.jpg', r)
                objectBase64 = base64.b64encode(buffer.tobytes())
                outObject.append(objectBase64)
                i = i + 1

            # 透過原圖辨識後，拿物件去搜尋
            for r in data['object_embedding']:
                objectEmbedding.append(r.tolist())
                oo = p_search(r)
                for rr in DataCollection(oo):
                    for rr2 in rr['pred']:
                        outObjectResult.append(rr2)

            # 原圖搜尋結果
            original = p_search(data['original_embedding'])
            originalEmbedding.append(data['original_embedding'].tolist())
            for rr in DataCollection(original):
                for rr2 in rr['pred']:
                    outOriginalResult.append(rr2)

        jsonObj['box'] = outBox
        jsonObj['class'] = outClass
        jsonObj['original_search'] = get_unique_list(outOriginalResult)
        jsonObj['object_search'] = get_unique_list(outObjectResult)
        jsonObj['object_file_base64'] = outObject
        jsonObj['original_file_base64'] = outOriginal
        jsonObj['object_embedding'] = objectEmbedding
        jsonObj['original_embedding'] = originalEmbedding
        return jsonObj
    except Exception as e:
        print(e)
        return {'status': False, 'msg': e}, 400


@app.post('/search')
async def search_images(image: UploadFile = None):
    try:
        if image is None:
            return {'status': False, 'msg': "image is None"}, 400

        content = await image.read()
        originalPath = UPLOAD_PATH + '/original-' + image.filename
        makedir(originalPath)
        with open(originalPath, "wb+") as f:
            f.write(content)

        res = p2(content)
        jsonObj = {}
        outClass = []
        outBox = []
        outObject = []
        outOriginal = []
        outObjectResult = []
        outOriginalResult = []
        objectEmbedding = []
        originalEmbedding = []
        for data in DataCollection(res):
            _, buffer = cv2.imencode('.jpg', data['original'])
            originalBase64 = base64.b64encode(buffer.tobytes())
            outOriginal.append(originalBase64)

            for r in data['class']:
                outClass.append(r)

            for r in data['box']:
                outBox.append(r)

            # 將圖片轉成base64
            i = 0
            for r in data['object']:
                objectPath = UPLOAD_PATH + '/' + data['class'][i] + '/' + data['class'][i] + '-object-' + image.filename
                makedir(objectPath)
                cv2.imwrite(objectPath, r)
                _, buffer = cv2.imencode('.jpg', r)
                objectBase64 = base64.b64encode(buffer.tobytes())
                outObject.append(objectBase64)
                i = i + 1

            # 透過原圖辨識後，拿物件去搜尋
            for r in data['object_embedding']:
                objectEmbedding.append(r.tolist())
                oo = p_search(r)
                for rr in DataCollection(oo):
                    for rr2 in rr['pred']:
                        outObjectResult.append(rr2)

            # 原圖搜尋結果
            original = p_search(data['original_embedding'])
            originalEmbedding.append(data['original_embedding'].tolist())
            for rr in DataCollection(original):
                for rr2 in rr['pred']:
                    outOriginalResult.append(rr2)

        jsonObj['box'] = outBox
        jsonObj['class'] = outClass
        jsonObj['original_search'] = get_unique_list(outOriginalResult)
        jsonObj['object_search'] = get_unique_list(outObjectResult)
        jsonObj['object_file_base64'] = outObject
        jsonObj['original_file_base64'] = outOriginal
        jsonObj['object_embedding'] = objectEmbedding
        jsonObj['original_embedding'] = originalEmbedding
        return jsonObj
    except Exception as e:
        print(e)
        return {'status': False, 'msg': e}, 400