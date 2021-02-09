import re
import json
from collections import defaultdict

pattern = r"^https://netzpolitik\.org/20[0-9][0-9]/.+[^#respond]$"


if __name__ == "__main__":

    keys = set()
    ref_count = 0
    id_count = 0

    with open('data/netzpolitik.jsonl', 'r', encoding="utf-8") as fi:
        for line in fi:
            js = json.loads(line)
            keys.add(js['id'])

    with open('data/netzpolitik.jsonl', 'r', encoding="utf-8") as read_fi:
        with open('data/judgement_list_netzpolitik.jsonl', 'w', encoding="utf-8") as out_fi:
            for line in read_fi:
                js = json.loads(line)
                try:
                    judgement = {
                        'id': js['id'],
                        'references': []
                    }

                    for ref in js['references']:
                        # make sure reference is present in dataset
                        if re.match(pattern, ref) and ref in keys:
                            ref_count += 1
                            judgement['references'].append(ref)

                    if len(judgement['references']) > 0:
                        id_count += 1
                        json.dump(judgement, out_fi)
                        out_fi.write('\n')

                except Exception:
                    print("error")
    
    print("docs count:", id_count)
    print("ref count:", ref_count)

# docs count: 7691
# ref count: 26290