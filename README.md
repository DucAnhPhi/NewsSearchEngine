# Background Linking of News Articles

This repository contains the code for my master thesis *Background Linking of News Articles*. This includes (installation) instructions and executable scripts, such that all conducted experiments, methods and results that are presented in my work can be reproduced or built up on.

- [Abstract](#abstract)
- [Installation](#installation)
- [Run experiments](#getting-started)
- [Run unit tests](#run-unit-tests)
- [Datasets](#datasets)
- [License](#license)
- [Author](#author)

## Abstract
The goal of the *Background Linking of News Articles* is to help a reader understand or learn more about the topic or main issues in the current article that they are reading. Given a news article, the task is to retrieve other news articles that provide important context or background information to the reader. In this work the background linking task is treated as a search problem that is divided into two steps: retrieval and ranking. Based on these two steps we develop an information retrieval (IR) framework in the form of a conceptual search pipeline that can be seen as a guideline, as it shows necessary components needed for search, without imposing specific methods or data structures. It comes with technical considerations and supports the use of multiple IR methods in parallel. On the basis of this framework we test two drastically different IR methods on the *TREC Washington Post Corpus* and on a self-scraped corpus, consisting of news articles from the German-speaking *Netzpolitik.org* news outlet. For the latter corpus we could show that internal hyperlinks placed by the author can be used as an heuristic for background links, which is effective for evaluating retrieval recall. The bespoke IR methods are the *ranked Boolean model*, which serves as a baseline method, and a *semantic search* approach based on *sentence-embeddings* and *Hierarchical Navigable Small World* (HNSW) graphs as the underlying index structure. Even with optimized sentence-embeddings, we could show that the semantic search approach is not effective for both the retrieval and ranking task, not even as a complementary approach, as a better recall-to-precision ratio can be achieved for the baseline method. Our baseline approach achieves an *nDCG@5* score of 0.5205 on the TREC 2020 test data. In comparison the best performing model (that was published at the time of this writing) among all *TREC News Track 2020* submissions for the background linking task achieves an *nDCG@5* score of 0.5924. In this context there is still room for improvement for the research community.

## Installation
* Install Python 3.6.9 (if not already installed)

* On Windows: [Install Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)

* Install [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

* **Recommended:**
Setup a python virtual environment for this project. It keeps the dependencies required by different projects in separate places.

```
$ pip install virtualenv

$ cd NewsSearchEngine

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
Before you can run the following scripts, you have to change directory to the parent directory of the current project:

```
python -m NewsSearchEngine.wapo.experiments.keyword_match_recall
python -m NewsSearchEngine.wapo.experiments.semantic_search_recall
python -m NewsSearchEngine.wapo.experiments.combined_recall
python -m NewsSearchEngine.wapo.experiments.ranking
```

```
python -m NewsSearchEngine.netzpolitik.experiments.keyword_match_recall
python -m NewsSearchEngine.netzpolitik.experiments.semantic_search_recall
python -m NewsSearchEngine.netzpolitik.experiments.combined_recall
```

## Datasets
The following two datasets are used for the experiments:

### TREC Washington Post document collection (WAPO)

The evaluation mainly relies on the data that TREC provided for the News Track Background Linking Task from 2018-2020. This data is made up of two components: the TREC Washington Post Collection (WAPO) and corresponding test data, that is, human-rated relevance judgments for various test topics:

- News Track 2018: WAPO v2; test data: [50 topics](./data/wapo_newsir18_topics.txt), [8,508 judgments in total](./data/wapo_newsir18_bqrels.txt)
- News Track 2019: WAPO v2; test data: [57 topics](./data/wapo_newsir19_topics.txt), [15,655 judgments in total](./data/wapo_newsir19_bqrels.txt)
- News Track 2020: WAPO v3; test data: [49 topics](./data/wapo_newsir20_topics.txt), [17,764 judgments in total](./data/wapo_newsir20_bqrels.txt)

The human-rated relevance values map to the following judgments \cite{TrecGuide}:

- 0: The document provides little or no useful background information.
- 2: The document provides some useful background or contextual information that would help the user understand the broader story context of the target article.
- 4: The document provides significantly useful background \ldots
- 8: The document provides essential useful background \ldots
- 16: The document \textit{must} appear in the sidebar otherwise critical context is missing.

The initial WAPO version (v1) contained duplicate entries with the same document identifier (*id* field in the JSON object.), which were removed in version 2.  That version still contained a number of near-duplicate documents which have been removed in v3, with the help of a near-duplicate detection system. Additionally, v3 adds 154,418 new documents from 2018 and 2019, in comparison to v2.

We choose to use WAPO v3 over previous versions, because it satisfies our assumption that there are close to no near-duplicate documents in the collection.


Originally, the WAPO v3 collection contains 671,947 news articles and blog posts from January 2012 through December 2019. We removed articles that are labeled in the *kicker* field as *Test*. These "test" articles only contain sample text, such as *"Lorem ipsum ..."*, and do not represent real articles. For this reason, they can be seen as noise. Furthermore, articles without a body are removed as well. In total we removed 4,179 articles.


The TREC news track in 2020 is based on the WAPO v3 collection, though previous tracks in 2018 and 2019 are based on version v2. We do not want to discard test data from previous tracks that contain articles only present in v2. At the same time, we want to use the superior version v3. As a compromise, we added 2,241 v2-articles that are referenced in previous test data and are not contained in v3 to the WAPO v3 collection. Finally, the cleaned and updated WAPO v3 collection contains **670,009** documents.


According to the TREC 2020 News Track Guidelines, articles from the dataset that are labeled in the *kicker* field as *Opinion*, *Letters to the Editor*, or *The Post's View*, are considered irrelevant by the human-raters. Initially, we made the mistake and removed these "irrelevant" articles as well, but later found out that they are referenced in the test data as negative examples. For this reason, it is not advisable to remove these articles and rather use the *kicker* field as a (negative) relevance signal for the ranking phase.

### Netzpolitik
The motivation behind including another dataset for the evaluation is to have more data diversity. This comes with additional implementation efforts and require more hardware resources, mainly memory- and computational-costs. We choose the German news outlet *Netzpolitik.org* for the following reasons:

- as it is a German outlet, we can explore and evaluate our framework for the German language as well, instead of only the English language. This is going to reveal whether our concept is applicable for the German language, and which configurations would be necessary.
- all of its articles are online accessible, without charge or restrictions, e.g., pay-walls or limited usage.
- its simple and predictable URL routing patterns make crawling articles easier.
- most articles contain links to other internal articles that support the content. These internal references are considered as background links and are needed for the recall evaluation (see Section \ref{sec:eval_retrieval}).
- most articles contain human-annotated keywords that sum up the content. These keywords can be used for retrieval processes.


For this outlet, we crawled all articles published from 2012-2020, which totals to **14,246** articles, ignoring articles without a title or body.

## Run unit tests

Run the following command in the current directory:

```
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Author

* **Phi, Duc Anh**
