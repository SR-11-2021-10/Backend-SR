FROM python:3.8

COPY . .

#RUN apk add build-base

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]