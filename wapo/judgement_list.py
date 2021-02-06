import os
import json

if __name__ == "__main__":
    topic_dict = {}
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    # generate topic dictionary
    with open(f"{data_location}/wapo_newsir18-topics.txt", "r") as f:
        for line in f:
            topic = line.strip().split(" ")
            topic_dict[topic[0]] = topic[1]
    # generate judgement list with translated topic ids
    with open(f"{data_location}/wapo_bqrels.exp-gains.txt", "r") as fin:
        with open(f"{data_location}/judgement_list_wapo.jsonl", "w") as fout:
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