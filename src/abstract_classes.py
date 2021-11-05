from abc import ABC, abstractmethod

class Vectorizer(ABC):

    important_topics = {
        "Narcotics": ["cannabis", "weed", "cocaine", "methamphetamine", "drugs", "marijuana", "MDMA", "LSD", "KETAMINA",
                      "heroin", "fentanyl", "Narcotics"],
        "Weapons": ["Weapons", "gun", "knife", "weapon", "firearm", "rifle", "carabine", "shotgun", "assaults", "sword",
                    "blunt"],
        "Investigation": ["Investigation", "gang", "mafia", "serial", "killer", "rape", "thefts", "recidivism", "arrest",
                          "ethnicity", "caucasian", "afroamerican", "hispanic", "robbery", "cybercrime"]
    }

    def __init__(self, documents, tokenizer):
        self.tokenizer = tokenizer
        self.documents = documents

    @abstractmethod
    def vectors(self):
        pass

    @abstractmethod
    def vec(self, document):
        pass

    @abstractmethod
    def increaseWeightImportantWords(self, vectors, multiplier=2):
        pass

class Tokenizer(ABC): 

    too_frequent_words = ['action', 'contract', 'defendants', 'plaintiff', 'trial', '2d', 'sentence', 'state', 'property',
                          'plaintiffs', 'case', 'appellee', 'made', 'upon', 'judgment', 'counsel', 'people',
                          'appellant', 'law', 'defendant', 'act', 'said']

    @abstractmethod
    def tokenize(self, text):
        pass
    