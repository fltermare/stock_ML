FROM tensorflow/tensorflow:2.3.1

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py /app/
ADD stocknet /app/stocknet
