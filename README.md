## TODO
- get strong baseline retrieval method with high recall
- enforce static type checking
- implement LTR
- support more outlets
- handle module requirements
- handle scaling vector storage (max elements)
- implement webapp with async component and model server
- log queries and clicks for judgement list and popularity metric


## Python Version

Python 3.6.9

## Run Unittests

Run the following command in the current directory:

```
pytest
```

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