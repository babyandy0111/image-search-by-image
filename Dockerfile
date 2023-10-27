FROM python:3.8.18
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./.env /code/.env
COPY ./api.py /code/api.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]