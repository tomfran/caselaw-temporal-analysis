import os
import json
import re
from collections import defaultdict

class Dataset(): 
    
    def __init__(self, 
                 dataset_path="", 
                 save_folder="../data/processed/docs", 
                 tokens_folder="../data/processed/tokens", 
                 topics_folder="../data/processed/topics"):
        self.dataset_path = dataset_path
        self.save_folder = save_folder
        self.tokens_folder = tokens_folder
        self.topics_folder = topics_folder
        
    def process_line(self, document):
        
        def clean(word):
            word = re.sub(r'[\W_]+', ' ', word)
            word = re.sub(r'\d+', '', word)
            return word.lower()
        
        to_save = {} 
        document = json.loads(document)
        
        to_save["id"] = document["id"]
        to_save["name"] = document["name"]
        to_save["decision_date"] = int(document["decision_date"][:4])
        to_save["court"] = document["court"]["name"]
        text = ""
        for op in document["casebody"]["data"]["opinions"]:
            text += op["text"] + " "
            
        to_save["text"] = clean(text)
        
        return to_save
    
    def process_lines(self):
        with open(self.dataset_path) as f:
            return [self.process_line(line) for line in f]
        
    def partition_data(self):
        partitions = defaultdict(lambda : [])
        
        data = self.process_lines()
        for case in sorted(data, key=lambda x : x["decision_date"]):
            year = case["decision_date"]//20 * 20
            partitions[year] += [case]
            
        return partitions
            
    def partition_save(self):
        path = self.save_folder
        partitions = self.partition_data()
        
        for year, documents in partitions.items():
            path = f"{self.save_folder}/{year}.json"
            print(f"Saving documents from {year} to {year+19}")
            with open(path, "w") as f:
                f.write(json.dumps(documents))

    def merge_tokens_topics_data(self):
        docs = sorted(os.listdir(self.save_folder))
        
        for doc in docs:
            doc_path = f"{self.save_folder}/{doc}"
            tokens_path = f"{self.tokens_folder}/{doc}"
            topics_path = f"{self.topics_folder}/{doc}"
            print(f"Processing {doc}")
            
            data = self.load_json(doc_path)
            tokens = self.load_json(tokens_path)
            topics = self.load_json(topics_path)
            
            for i,el in enumerate(data):
                el["tokens"] = tokens[i]
                el["topic"] = [float(e) for e in topics[i]]
            
            with open(doc_path, "w") as f:
                f.write(json.dumps(data))
            
    def load_json(self, path): 
        with open(path, "r") as f: 
            return json.load(f)
        
    def _filter_dict(self, d, fields):
        return { k : v for k, v in d.items() if k in fields}

    def load_dataset(self, year=None, fields=None, courts=None):
        
        if year:
            data = self.load_json(f"{self.save_folder}/{year}.json")
        else:
            file_names = [f"{self.save_folder}/{file}" for file in sorted(os.listdir(self.save_folder))]    
            data = []
            for f in file_names:
                data += self.load_json(f)
                
        if courts:
            data = [el for el in data if el["court"] in courts]
        
        if fields:
            return [self._filter_dict(el, fields) for el in data]
            
        return data