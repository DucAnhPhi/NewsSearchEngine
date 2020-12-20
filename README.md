
- [TODOs](#todos)
- [Installation](#installation)
- [Getting started](#getting-started)
- [License](#license)
- [Author](#author)

## TODOs
- get strong baseline retrieval method with high recall
- enforce static type checking
- implement LTR
- support more outlets
- handle module requirements
- handle scaling vector storage (max elements)
- implement webapp with async component and model server
- log queries and clicks for judgement list and popularity metric

## Installation
* Install Python 3.6.9 (if not already installed)

* **Recommended:**
Setup a python virtual environment for this project. It keeps the dependencies required by different projects in separate places.

```
$ pip install virtualenv

$ cd LinguisticAnalysis

$ virtualenv -p python3.6.9 env
```
* To begin using the virtual environment, it needs to be activated:

```
$ source env/bin/activate
```

* Finally install all dependencies running:

```
$ pip install requirements.txt
```

* If you are done working in the virtual environment for the moment, you can deactivate it (remember to activate it again on usage):

```
$ deactivate
```

* To delete a virtual environment, just delete its folder.

```
rm -rf env
```

## Getting started
### Run Unittests

Run the following command in the current directory:

```
pytest
```

### Run News Scraper

Run the following command to run the scraper and save the output to **data/netzpolitik.jsonl**:

```
scrapy runspider scraper_netzpolitik.py
```

### Run Embedding API

Run the following command to run the embedding API:

```
$ export FLASK_APP=embedding/api.py
$ python -m flask run
 * Running on http://127.0.0.1:5000/
```

### Request Embedding API

When the Embedding API is running you can access the embedding functionality via HTTP GET requests:

```
curl --location --request GET 'http://localhost:5000' \
--form 'data="your string"'
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Author

* **Phi, Duc Anh**