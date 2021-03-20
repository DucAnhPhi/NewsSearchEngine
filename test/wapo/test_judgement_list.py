import pytest
import os
from ...wapo.judgement_list import JudgementListWapo

class TestJudgementListWAPO():
    def test_create_judgement_list(self):
        actual = JudgementListWapo.get_topic_dict('18')
        expected = {
            "321": "9171debc316e5e2782e0d2404ca7d09d",
            "336": "2a340b8573d498e261d6f2365b37f8eb",
            "341": "7ef8ce1720bf2f6b2065a97506ee89b4",
            "347": "c3cea789141ef2ae856419e86e165e0c",
            "350": "985b90cc-7c98-11e3-93c1-0e888170b723",
            "362": "4989ebfeb752e6b317d1ef3997b21a01",
            "363": "474ae088-ab1e-11e4-9c91-e9d2f9fde644",
            "367": "1e03fecf4d33b7896203298ab3858156",
            "375": "0e85b0c0-f7ef-11e4-9030-b4732caefe81",
            "378": "3c5be31e-24ab-11e5-b621-b55e495e9b78",
            "393": "fef0f232a9bd94bdb96bac48c7705503",
            "397": "563fb77e-024f-11e6-9203-7b8670959b88",
            "400": "72e72b41097d53b627fd375dd2d3309b",
            "408": "988147454a2b8eafd1535cd673dd04ba",
            "414": "4192b016-8708-11e3-a5bd-844629433ba3",
            "422": "145b9a6caa16d931c108a89798e65e17",
            "426": "56f0438ee0fb34c341ccf5af36de5175",
            "427": "2e83ad87eb1bade22e6e96ece616c24f",
            "433": "159e6f9e-8e84-11e3-84e1-27626c5ef5fb",
            "439": "5c466d4a01492f1b5cc9758e19429a1f",
            "442": "3902c9005a0563742fc4acb2c011b164",
            "445": "c8351276-76de-41f1-b294-4f3e5d373c8c",
            "626": "a79b1b7d8cc5273d4995fec5e122e44b",
            "646": "6fdc62d37aaf685b809c501abe13c56c",
            "690": "defd7f4a85496d52a210938d58a7ae76",
            "801": "b0235f56-1cce-11e4-ae54-0cfe1f974f8a",
            "802": "6668d83480f5c58b54a90770835ac2d4",
            "803": "cad56e871cd0bca6cc77e97ffe246258",
            "804": "579e9ae8-6a2f-11e6-8225-fbb8a6fc65bc",
            "805": "5ec40b6bc6c5f4487132da7be04fc914",
            "806": "2bea9433d4e1050c9c85175df466b3e2",
            "807": "11915bd8-7944-11e2-9c27-fdd594ea6286",
            "808": "30a493b8-fb07-11e4-9ef4-1bb7ce3b3fb7",
            "809": "02e52bdba097c9df4cbae66e04f82542",
            "810": "9dd7b85cd1e3da1b5c8e79f32fec7177",
            "811": "a244d1e0cfd916a2af76b6a6c785b017",
            "812": "dcd1560bd13a0b665b95d3ba27cc960c",
            "813": "b4c6361974466458bb721b9b1628220b",
            "814": "e1336b8f-b0c2-4610-9a3c-ec85a546c9ad",
            "815": "a36fa8a2-8962-11e6-bff0-d53f592f176e",
            "816": "37a8e2283e4677b703f6464d0191a700",
            "817": "bd1e6cc8d7525fec36a717be45638bf4",
            "818": "a2744bb98e1968307548e4976232cf1c",
            "819": "5f37aac53768e749b861028397eb6849",
            "820": "fc1ca759c9c433376e71884870d225ab",
            "821": "c6bf4a4bf542b7c67987c222d73def4b",
            "822": "43e9f3f12982c0e0bb15ad64b33a89c0",
            "823": "c109cc839f2d2414251471c48ae5515c",
            "824": "30c00b60-13f6-11e3-b182-1b3bb2eb474c",
            "825": "a1c41a70-35c7-11e3-8a0e-4e2cf80831fc"
        }
        assert actual == expected

    def test_create_judgement_list_unique(self):
        jl = JudgementListWapo.create(["18", "19", "20"],test=True)
        keys = set()
        for key in jl:
            keys.add(key)
        assert len(keys) == len(jl)

    def test_missing_articles_unique(self):
        missing_articles = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/wapo_missing_articles.jsonl"
        ids = set()
        count = 0
        with open(missing_articles, "r", encoding="utf-8") as f:
            for line in f:
                count += 1
                article_id = line.strip()
                ids.add(article_id)
        assert len(ids) == count