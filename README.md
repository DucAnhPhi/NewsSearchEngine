## Python Version

Python 3.6.9


## Run News Scraper

Run the following command to run the scraper and save the output to **data/netzpolitik.jsonl**:

```
scrapy runspider scraper_netzpolitik.py
```

## Run Embedding API

Run the following command to run the embedding API:

```
$ export FLASK_APP=embedding/api.py
$ python -m flask run
 * Running on http://127.0.0.1:5000/
```

## Request Embedding API

When the Embedding API is running you can access the embedding functionality via HTTP GET requests:

```
curl --location --request GET 'http://localhost:5000' \
--form 'data="your string"'
```