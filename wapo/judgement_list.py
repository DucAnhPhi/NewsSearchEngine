import os
import json
import re

class JudgementListWapo():
    @staticmethod
    def get_topic_dict():
        path = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data/wapo_newsir18-topics.txt"
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
    def create():
        data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
        topic_dict = JudgementListWapo.get_topic_dict()
        # generate judgement list with translated topic ids
        with open(f"{data_location}/wapo_bqrels.exp-gains.txt", "r", encoding="utf-8") as fin:
            with open(f"{data_location}/judgement_list_wapo.jsonl", "w", encoding="utf-8") as fout:
                jl = {}
                for line in fin:
                    orig_judg = line.strip().split(" ")
                    curr_id = topic_dict[orig_judg[0]]
                    if curr_id not in jl:
                        jl[curr_id] = [{"id": orig_judg[2], "exp_rel": orig_judg[3]}]
                        continue
                    jl[curr_id].append({"id": orig_judg[2], "exp_rel": orig_judg[3]})

                for key in jl:
                    judgement = {"id": key, "references": jl[key]}
                    json.dump(judgement, fout)
                    fout.write("\n")

if __name__ == "__main__":
    JudgementListWapo.create()