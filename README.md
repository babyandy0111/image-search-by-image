# image-search

# 用圖搜尋圖片
- `docker-compose -f docker-compose up -d`
- `pip3 install towhee==1.1.1 "fastapi[all]" uvicorn opencv-python-headless nest_asyncio==1.5.8 pymilvus==2.3.1 python-dotenv`
- 確認config.py裡的參數
- `python3 api.py`
- http://0.0.0.0:8000/docs
  - 確認doc

Note: 透過api取得圖片特徵值，並塞入向量資料庫進行查詢
