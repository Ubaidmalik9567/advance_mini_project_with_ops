FROM python:3.9

WORKDIR /app

COPY . /app/ 

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]
# The . (dot) represents the current directory 