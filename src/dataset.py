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
        
    def save_json(self, processed):
        with open(self.save_path, "w") as f:
            f.write(json.dumps(processed))
            
    def load_json(self): 
        with open(self.save_path, "r") as f: 
            return json.load(f)
        
    def load_text_list(self, size=-1):
        data = self.load_json()
        texts = [document["opinions"][i]["text"] 
                for document in data 
                for i in range(len(document["opinions"]))][::size] 
        return texts[0:size] if size != -1 else texts