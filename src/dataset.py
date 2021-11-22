import os
import json
import re
from collections import defaultdict

class Dataset(): 
    
    def __init__(self, 
                 dataset_path="", 
                 save_folder="../data/processed/docs", 
                 tokens_folder="../data/processed/tokens"):
        self.dataset_path = dataset_path
        self.save_folder = save_folder
        self.tokens_folder = tokens_folder
        
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
            
    def load_json(self, path): 
        with open(path, "r") as f: 
            return json.load(f)
        
    def load_dataset(self, year=None, tokens=False):
        base_folder = self.tokens_folder if tokens else self.save_folder
        
        if year:
            return self.load_json(f"{base_folder}/{year}.json")
        
        file_names = [f"{base_folder}/{file}" for file in sorted(os.listdir(base_folder))]
        
        data = []
        for f in file_names:
            data += self.load_json(f)
        
        return data