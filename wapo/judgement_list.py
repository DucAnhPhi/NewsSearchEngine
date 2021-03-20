import argparse
import json
import os
import re
from elasticsearch import Elasticsearch

class JudgementListWapo():
    @staticmethod
    def get_topic_dict(year:str):
        path = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data/wapo_newsir{year}_topics.txt"
        topic_dict = {}
        # read topic dictionary
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            topics = re.findall(r"\s[0-9]{3}\s", data)
            topics = [t.strip() for t in topics]
            doc_ids = re.findall(r"<docid>[\w-]+", data)
            doc_ids = [i[7:] for i in doc_ids]
            for i,topic in enumerate(topics):
                topic_dict[topic] = doc_ids[i]
        return topic_dict

    @staticmethod
    def create(years, filename="judgement_list_wapo", test=False):
        data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
        topic_combined = {}
        jl = {}
        for year in years:
            topic_dict = JudgementListWapo.get_topic_dict(year)
            # generate judgement list with translated topic ids
            with open(f"{data_location}/wapo_newsir{year}_bqrels.txt", "r", encoding="utf-8") as fin:
                for line in fin:
                    orig_judg = line.strip().split(" ")
                    curr_id = topic_dict[orig_judg[0]]
                    if curr_id not in jl:
                        jl[curr_id] = [{"id": orig_judg[2], "exp_rel": orig_judg[3]}]
                        continue
                    jl[curr_id].append({"id": orig_judg[2], "exp_rel": orig_judg[3]})

        if test:
            return jl
        with open(f"{data_location}/{filename}.jsonl", "w", encoding="utf-8") as fout:
            for key in jl:
                judgement = {"id": key, "references": jl[key]}
                json.dump(judgement, fout)
                fout.write("\n")
        print("created wapo judgment list.")

    @staticmethod
    def examine(es):
        def is_not_relevant(kicker: str):
            is_not_relevant = False
            if kicker:
                not_relevant = {
                    #"test",
                    "opinion",
                    "letters to the editor",
                    "the post's view"
                }
                is_not_relevant = kicker.lower() in not_relevant
            return is_not_relevant
        data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
        years = ["20"]
        topic_combined = {}
        not_relevant_count = 0
        exception_count = 0
        opinion = {"0": 0,"2": 0, "4": 0, "8": 0, "16": 0}
        letters = {"0": 0,"2": 0, "4": 0, "8": 0, "16": 0}
        posts = {"0": 0,"2": 0, "4": 0, "8": 0, "16": 0}
        no_title = {"0": 0,"2": 0, "4": 0, "8": 0, "16": 0}
        no_text = {"0": 0,"2": 0, "4": 0, "8": 0, "16": 0}
        for year in years:
            topic_dict = JudgementListWapo.get_topic_dict(year)
            # generate judgement list with translated topic ids
            with open(f"{data_location}/wapo_newsir{year}_bqrels.txt", "r", encoding="utf-8") as fin:
                for line in fin:
                    orig_judg = line.strip().split(" ")
                    curr_id = topic_dict[orig_judg[0]]
                    if es:
                        try:
                            es.get(
                                index = "wapo_clean",
                                id = curr_id
                            )
                            if int(orig_judg[3]) > 0:
                                es.get(
                                    index = "wapo_clean",
                                    id = orig_judg[2]
                                )
                        except Exception as e:
                            try:
                                orig = es.get(
                                    index = "wapo",
                                    id = orig_judg[2]
                                )["_source"]
                                kicker = orig["kicker"]
                                if is_not_relevant(kicker):
                                    if kicker.lower() == "opinion":
                                        opinion[orig_judg[3]] += 1
                                    if kicker.lower() == "letters to the editor":
                                        letters[orig_judg[3]] += 1
                                    if kicker.lower() == "the post's view":
                                        posts[orig_judg[3]] += 1
                                    not_relevant_count += 1
                                else:
                                    if not orig["title"]:
                                        no_title[orig_judg[3]] += 1
                                    if not orig["text"]:
                                        no_text[orig_judg[3]] += 1
                                exception_count += 1
                            except Exception as e:
                                print(e)
                            continue

        print(f"opinion: {opinion}")
        print(f"letters: {letters}")
        print(f"posts: {posts}")
        print(f"no title: {no_title}")
        print(f"no text: {no_text}")
        print(f"total not found: {exception_count}")
        print(f"not found and declared irrelevant: {not_relevant_count}")

if __name__ == "__main__":
    index = "wapo_clean"
    p = argparse.ArgumentParser(description='Generate WAPO judgement list')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default=index, help='index name')
    p.add_argument('--user', default=None, help='ElasticSearch user')
    p.add_argument('--secret', default=None, help="ElasticSearch secret")

    args = p.parse_args()

    es = None

    if args.user and args.secret:
        es = Elasticsearch(
            hosts = [{"host": args.host, "port": args.port}],
            http_auth=(args.user, args.secret),
            scheme="https",
            retry_on_timeout=True,
            max_retries=10
        )
    else:
        es = Elasticsearch(
            hosts=[{"host": args.host, "port": args.port}],
            retry_on_timeout=True,
            max_retries=10
        )

    JudgementListWapo.examine(es)
    #JudgementListWapo.create(["18","19","20"], "judgement_list_wapo_combined.jsonl")
    #JudgementListWapo.create(["18"], "judgement_list_wapo_18.jsonl")
    #JudgementListWapo.create(["19"], "judgement_list_wapo_19.jsonl")
    #JudgementListWapo.create(["20"], "judgement_list_wapo_20.jsonl")