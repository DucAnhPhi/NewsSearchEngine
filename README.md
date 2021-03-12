
- [TODOs](#todos)
- [Installation](#installation)
- [Run experiments](#getting-started)
- [Run unit tests](#run-unit-tests)
- [License](#license)
- [Author](#author)

## TODOs
- implement filtering
- implement LTR
- handle scaling vector storage (max elements)

## Installation
* Install Python 3.6.9 (if not already installed)

* On Windows: [Install Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)

* Install [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

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
$ pip install -r requirements.txt
```

* If you are done working in the virtual environment for the moment, you can deactivate it (remember to activate it again on usage):

```
$ deactivate
```

* To delete a virtual environment, just delete its folder.

```
rm -rf env
```

### Speed up performance

You can speed up the encoding of embeddings if your machine has a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with atleast 2GB of free video memory:
- Install [CUDA](https://developer.nvidia.com/cuda-downloads)

Please reinstall pytorch with the your current CUDA version selected (also select Pip as the Package) [here](https://pytorch.org/get-started/locally/)


## Run experiments




### Download Datasets

The following datasets were used to evaluate all experiments in this repository. Please download the following datasets if you want to reproduce our experiments:

- To reproduce our experiments with the *TREC Washington Post Corpus (WAPO)*, request the dataset [here](https://trec.nist.gov/data/wapost/) and store the **.jl** file in the **data/** folder
- To reproduce our experiments with *netzpolitik.org* run the following command in the parent directory of the project, to scrape all articles from 2012 to 2020 (the output is going to be stored in **data/netzpolitik.jsonl**):
```
python -m NewsSearchEngine.netzpolitik.scraper
```

### Index Datasets

We provide indexing scripts for both datasets. Go to the parent directory of this project and run the following commands:

```
python -m NewsSearchEngine.wapo.index_es
python -m NewsSearchEngine.wapo.index_vs
```


```
python -m NewsSearchEngine.netzpolitik.index_es
```

### Run experiment scripts

```
python -m NewsSearchEngine.wapo.experiments.keyword_match_recall
```

```
python -m NewsSearchEngine.netzpolitik.experiments.keyword_match_recall

python -m NewsSearchEngine.netzpolitik.experiments.semantic_search_recall

python -m NewsSearchEngine.netzpolitik.experiments.combined_recall
```

### TREC Washington Post document collection (WAPO)

The TREC Washington Post Corpus contains 671,947 news articles and blog posts from January 2012 through December 2019. You can request this data [here](https://trec.nist.gov/data/wapost/).

For the background linking task in 2018 (TREC 2018), TREC provided:

- [50 test topics for background linking task](https://trec.nist.gov/data/news/2018/newsir18-topics.txt)
- [relevance judgements for backgroundlinking task (with exponential gain values)](https://trec.nist.gov/data/news/2018/bqrels.exp-gains.txt)

The relevance values map to the following judgments:


- 0: The document provides little or no useful background information.
- 2: The document provides some useful background or contextual information that would help the user understand the broader story context of the target article.
- 4: The document provides significantly useful background ...
- 8: The document provides essential useful background ...
- 16: The document _must_ appear in the sidebar otherwise critical context is missing.

We put both sets together in a more usable [JSON Lines Format](https://jsonlines.org/) in **data/judgement_list_wapo.jsonl** by using the following script: **wapo/judgement_list.py**.

According to the [TREC 2020 News Track Guidelines](http://trec-news.org/guidelines-2020.pdf) we removed articles from the dataset which are labeled in the "kicker" field as "Opinion", "Letters to the Editor", or "The Post's View", as they are **not relevant**. Additionally we removed articles which are labeled in the "kicker" field as "Test" as they contain "Lorem ipsum" text, thus being irrelevant aswell.
After the filtering 629381 news articles remain, which is a decrease in size by 6.3%.


## Run unit tests

Run the following command in the current directory:

```
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Author

* **Phi, Duc Anh**