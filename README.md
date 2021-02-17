
- [TODOs](#todos)
- [Installation](#installation)
- [Run experiments](#getting-started)
- [Run unit tests](#run-unit-tests)
- [License](#license)
- [Author](#author)

## TODOs
- get strong baseline retrieval method with high recall
- implement LTR
- handle scaling vector storage (max elements)
- support more outlets

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

We put both sets together in a more usable [JSON Lines Format](https://jsonlines.org/) in **data/judgement_list_wapo.jsonl** by using the following script: **wapo/judgement_list.py**.

According to the [TREC 2020 News Track Guidelines](http://trec-news.org/guidelines-2020.pdf) we removed articles from the dataset which are labeled in the "kicker" field as "Opinion", "Letters to the Editor", or "The Post's View", as they are **not relevant**. Additionally we removed articles which are labeled in the "kicker" field as "Test" as they contain "Lorem ipsum" text, thus being irrelevant aswell.
After the filtering only 487,322 news articles remain, which is a decrease in size by 27%.

It is important to note, that 5 articles listed in the [50 test topics for background linking task](https://trec.nist.gov/data/news/2018/newsir18-topics.txt) are missing in the dataset:

```
# available
<top>
<num> Number: 321 </num>
<docid>9171debc316e5e2782e0d2404ca7d09d</docid>
<url>https://www.washingtonpost.com/news/worldviews/wp/2016/09/01/women-are-half-of-the-world-but-only-22-percent-of-its-parliaments/<url>
</top>

# not available
<top>
<num> Number: 823 </num>
<docid>c109cc839f2d2414251471c48ae5515c</docid>
<url>https://www.washingtonpost.com/news/to-your-health/wp/2016/09/21/superbug-mrsa-may-be-spreading-through-tainted-poultry/2695269178/</url>
</top>

# available
<num> Number: 812 </num>
<docid>dcd1560bd13a0b665b95d3ba27cc960c</docid>
<url>https://www.washingtonpost.com/news/morning-mix/wp/2016/05/23/after-years-of-alleged-bullying-an-ohio-teen-killed-herself-is-her-school-district-responsible/<url>

# available
<num> Number: 811 </num>
<docid>a244d1e0cfd916a2af76b6a6c785b017</docid>
<url>https://www.washingtonpost.com/news/morning-mix/wp/2015/07/22/car-hacking-just-got-real-hackers-disable-suv-on-busy-highway/</url>

# not available
<num> Number: 803 </num>
<docid>cad56e871cd0bca6cc77e97ffe246258</docid>
<url>https://www.washingtonpost.com/news/wonk/wp/2016/05/11/the-middle-class-is-shrinking-just-about-everywhere-in-america/4/</url>
```

We added the 3 of the missing but available articles in the **wapo/index_es.py** script.


## Run unit tests

Run the following command in the current directory:

```
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Author

* **Phi, Duc Anh**