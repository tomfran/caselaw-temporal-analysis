import json

class Dataset(): 
    
    def __init__(self, dataset_path, save_path):
        self.dataset_path = dataset_path
        self.save_path = save_path
    
    def process_line(self, document):
        to_save = {} 
        document = json.loads(document)
        
        to_save["id"] = document["id"]
        to_save["name"] = document["name"]
        to_save["decision_date"] = int(document["decision_date"][:4])
        to_save["opinions"] = document["casebody"]["data"]["opinions"]
        return to_save         
    
    def process_lines(self):
        with open(self.dataset_path) as f:
            return [self.process_line(line) for line in f]
        
    def save_json(self, processed, overwrite_path=None):
        
        path = self.save_path
        if overwrite_path:
            path = overwrite_path
                            
        with open(path, "w") as f:
            f.write(json.dumps(processed))
            
    def load_json(self): 
        with open(self.save_path, "r") as f: 
            return json.load(f)
        
    def load_text_list(self, size=-1, field_name="text"):
        data = self.load_json()
        texts = [document["opinions"][i][field_name] 
                for document in data 
                for i in range(len(document["opinions"]))]
        return texts[0:size] if size != -1 else texts
    
    def save_token_dataset(self, tokens_path="../data/processed/tokens.json"):
        tokens = json.load(open(tokens_path))
        data = self.load_json()
        
        i = 0
        for case in data:
            for opinion in case["opinions"]:
                opinion.pop("text")
                opinion["tokens"] = tokens[i]
                i += 1
                
        self.save_json(data, overwrite_path="../data/processed/tokenized_processed.json")