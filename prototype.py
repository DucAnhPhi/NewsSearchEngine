from vector_storage import VectorStorage
import pprint
import os.path

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    storage_location = 'data/storage.bin'
    if os.path.isfile(storage_location):
        print("Loading vector storage from file...\n")
        vs = VectorStorage(storage_location)
    else:
        print("Initialize vector storage.\n")
        vs = VectorStorage()
        print("Add items from file...\n")
        vs.add_items_from_file('data/netzpolitik.jsonl')

    while True:
        data = input("Get news based on your text: \n")
        recs = vs.get_k_nearest([data], 5)
        print("-----------------------------------------")
        print("Your recommendations: \n")
        pp.pprint(recs[0])
        print("-----------------------------------------")