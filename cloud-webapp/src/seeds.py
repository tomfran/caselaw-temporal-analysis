import os
import json


class Seeds:

    start_narcotics = {'cannabis', 'cocaine', 'methamphetamine', 'drugs', 'drug', 'marijuana',
                       'ecstasy', 'lsd', 'ketamine', 'heroin', 'fentanyl', 'overdose', 'blunt'}
    start_weapons = {'gun', 'knife', 'weapon', 'firearm', 'rifle', 'carbine', 'shotgun', 'handgun',
                     'revolver', 'musket', 'pistol', 'derringer', 'assault', 'rifle', 'sword'}
    start_investigation = {'gang', 'mafia', 'serial', 'killer', 'rape', 'theft', 'recidivism',
                           'arrest', 'robbery', 'cybercrime', 'cyber', 'crime'}

    def __init__(self,
                 path="../data/processed/seeds/"):
        self.path = path

    def get_starting_seeds(self):
        return self.start_narcotics, self.start_weapons, self.start_investigation

    def get_word2vec_seeds(self, filename="word2vec.json"):
        return self.get_seeds_from_file(filename)

    def get_word2vec_lemmatized_seeds(self, filename="w2v_lemmatized.json"):
        return self.get_seeds_from_file(filename)

    def get_word2vec_combined_seeds(self, filename="w2v_combined.json"):
        return self.get_seeds_from_file(filename)

    def get_final_filtered_seeds(self, filename="final_filtered.json"):
        return self.get_seeds_from_file(filename)

    def get_seeds_from_file(self, filename):
        with open(self.path + filename, "r") as f:
            data = json.load(f)
            return set(data["narcotics"]), set(data["weapons"]), set(data["investigation"])

    def save_word2vec_seeds(self, narcotics, weapons, investigation, filename="word2vec.json"):
        return self.save_seeds_to_file(narcotics, weapons, investigation, filename)

    def save_word2vec_lemmatized_seeds(self, narcotics, weapons, investigation, filename="w2v_lemmatized.json"):
        return self.save_seeds_to_file(narcotics, weapons, investigation, filename)

    def save_word2vec_combined_seeds(self, narcotics, weapons, investigation, filename="w2v_combined.json"):
        return self.save_seeds_to_file(narcotics, weapons, investigation, filename)

    def save_final_filtered_seeds(self, narcotics, weapons, investigation, filename="final_filtered.json"):
        return self.save_seeds_to_file(narcotics, weapons, investigation, filename)

    def save_seeds_to_file(self, narcotics, weapons, investigation, filename):
        data = {
            "narcotics": list(narcotics),
            "weapons": list(weapons),
            "investigation": list(investigation)
        }
        with open(self.path + filename, "w") as f:
            f.write(json.dumps(data))

    not_narcotics = {
        'suspect', 'medicine', 'gans', 'lab', 'dispensary', 'pain', 'sleeping', 'mouse', 'alliance',
        'asphyxia', 'cheap', 'allergic', 'mrs',
        'arrest', 'sleep', 'drowning', 'vertex', 'contain', 'labs', 'heart', 'acute', 'buy', 'magic',
        'pneumonia', 'usa', 'control', 'swallow',
        'abstracting', 'residue', 'treat', 'accidental', 'allergy', 'medications', 'dug', 'drown',
        'cooks', 'nonfatal', 'quaid', 'grow', 'c',
        'salt', 'highly', 'traffic', 'medicinal', 'dope', 'decriminalize', 'medication', 'manufacture',
        'reservation', 'precursor', 'plc', 'cook',
        'e', 'hepatitis', 'euphoric', 'boot', 'med', 'lace', 'lethal', 'twin', 'accidentally',
        'generic', 'coroner', 'malarial', 'restraint', 'ba',
        'controlled', 'ultram', 'powerful', 'fit', 'mood', 'danny', 'illicit', 'toxicity', 'effects',
        'commonly', 'his', 'quoad', 'n', 'legalize',
        'parkinson', 'fatal', 'amy', 'muscle', 'banning', 'sedentary', 'induce', 'electricity',
        'patch', 'horse', 'smell', 'reaction', 'sophisticated',
        'ingest', 'cardiac', 'mice', 'bath', 'swallowed', 'anti', 'legalizing', 'altering', 'mmj',
        'abstract', 'imitation', 'tar', 'unk', 'maker',
        'fatally', 'induced', 'medi', 'cocktail', 'surgical', 'grown', 'effect', 'patricia', 'co',
        'ban', 'quantity', 'ya', 'clandestine', 'entertainer',
        'new', 'animal', 'chemicals', 'storefront', 'pseudo', 'cr', 'containing', 'alias', 'newer',
        'boots', 'alter', 'delirium', 'asphyxiation', 'gamma', 'wholesaler',
        'bousfield', 'gastroscopy', 'ingesting', 'lod', 'patches', 'unichem'
    }
    not_weapons = {
        'addictive', 'bloodstaine', 'grievous', 'model', 'millimeter', 'instrument', 'plate', 'flame',
        'shoplift', 'm', 'disorderly', 'kitchen',
        'pastry', 'blender', 'plated', 'drug', 'pons', 'small', 'rape', 'power', 'battery', 'criminal',
        'uttering', 'arrest', 'attempt', 'bread',
        'bone', 'bar', 'cannabis', 'maggard', 'inch', 'fish', 'bloodstained', 'nonlethal', 'hang',
        'blacksmith', 'unregistere', 'chuck', 'reckless',
        'control', 'polydrug', 'grevious', 'revolutionary', 'chucks', 'utter', 'chrome', 'using',
        'crystal', 'treat', 'fillet', 'grab', 'tablets',
        'lewd', 'banta', 'harassment', 'fix', 'heroin', 'sandal', 'opiate', 'unlicensed', 'semi',
        'gravity', 'cause', 'article', 'salt', 'carrying',
        'recklessly', 'narcotic', 'highly', 'bodily', 'petit', 'dope', 'medicinal', 'prize',
        'decriminalize', 'toothbrush', 'carving', 'resist',
        'tablet', 'threats', 'vicodin', 'pen', 'felonious', 'occasion', 'crow', 'liberties', 'inflict',
        'wield', 'affray', 'automatic', 'charge',
        'machine', 'acto', 'x', 'harm', 'light', 'concealed', 'lace', 'forcible', 'stimulant',
        'springfield', 'prosecutor', 'cocaine',
        'methamphetamine', 'contin', 'custodial', 'powered', 'altercation', 'trespassing', 'mm', 'theft',
        'load', 'attempe', 'controlled', 'butcher',
        'attack', 'illicit', 'nun', 'steroid', 'brawl', 'possessing', 'double', 'facsimile', 'threat',
        'nosed', 'legalize', 'mischief', 'enactment',
        'officer', 'liberty', 'fantasy', 'caleb', 'paring', 'short', 'meat', 'green', 'crack',
        'tactical', 'incident', 'juana', 'fatal', 'unlawful',
        'vicious', 'intoxication', 'on', 'marihuana', 're', 'squad', 'banning', 'carry', 'civil',
        'peacock', 'nine', 'peeler', 'inflicting', 'stick',
        'synthetic', 'pon', 'leather', 'drugs', 'trespass', 'conceal', 'ceremonial', 'conduct',
        'cannabi', 'encampment', 'cultivation', 'flick',
        'confinement', 'aggravated', 'homemade', 'cock', 'h', 'epic', 'painkiller', 'bath',
        'amphetamines', 'flash', 'vegetable', 'sorcery', 'first',
        'ornamental', 'shoplifting', 'sexual', 'scissor', 'mallet', 'fife', 'unlicence', 'retractable',
        'malicious', 'imitation', 'amphetamine',
        'wrench', 'substances', 'steak', 'mighty', 'substance', 'burglary', 'centimeter', 'brittany',
        'serpent', 'stanley', 'narcotics', 'causing',
        'endangerment', 'addiction', 'sig', 'unlawfully', 'marijuana', 'bugle', 'ons', 'balls', 'forge',
        'assualt', 'domestic', 'felony', 'resisting',
        'curve', 'ball', 'unlicense', 'pare', 'stimulants', 'methadone', 'caine', 'ban', 'concealable',
        'class', 'co', 'deadly', 'meth', 'prescription',
        'sault', 'murder', 'allegedly', 'unprovoked', 'indecent', 'filet', 'box', 'hallucinogen',
        'potent', 'era', 'cobra', 'robbery', 'unregistered',
        'use', 'war', 'mari', 'molestation', 'salute', 'lascivious', 'incense', 'threatening',
        'threaten', 'degree', 'curved', 'phencyclidine',
        'possess', 'prescribe', 'warrior', 'abercrombie', 'bowie', 'prescribing', 'misdemeanor', 'fixed',
        'carve', 'varmint', 'subds', 'tresspass', 'xd', 'unlicenced', 'sticks', 'enactors', 'boning',
        'flaming', 'scepter', 'enactor', 'prized', 'insa', 'mightier', 'premediated', 'terroristic', 'sawn'
    }
    not_investigation = {
        'cache', 'drink', 'mask', 'scrap', 'id', 'protective', 'rest', 'surrey', 'prison', 'heart',
        'brega', 'firearm', 'bar', 'arousal',
        'gunman', 'touch', 'accuse', 'handgun', 'embezzlement', 'psychopath', 'utter',
        'misdemeanants', 'name', 'shooters', 'pizza',
        'truancy', 'mara', 'abortion', 'relapse', 'traffic', 'hate', 'carlo', 'incest',
        'concealing', 'jewelry', 'wield', 'sheet',
        'joshua', 'organize', 'task', 'dotson', 'prosecutor', 'parental', 'psychopathic',
        'daylight', 'banger', 'magistrate', 'offenses',
        'detaining', 'milly', 'nicola', 'attack', 'wielding', 'santos', 'parolees', 'facilities',
        'grand', 'organized', 'foxen',
        'postpone', 'hillside', 'in', 'arpaio', 'civilly', 'whereabouts', 'hilton', 'calculating',
        'proceedings', 'reoffende', 'illinois',
        'fedex', 'anton', 'bars', 'wnt', 'markus', 'cobras', 'debit', 'mccarthy', 'attacks',
        'behavior', 'reputed', 'tool', 'ralph',
        'subway', 'standoff', 'apts', 'bag', 'tamper', 'patronizing', 'ted', 'fake', 'felony',
        'oakdale', 'brotherhood', 'ms', 'scheme',
        'attorney', 'night', 'moat', 'unprotected', 'bully', 'mathew', 'selby', 'rates', 'bail',
        'stab', 'indecent', 'cell', 'kiddie',
        'detention', 'w', 'releasing', 'motorcycle', 'detainer', 'correctional', 'protection',
        'angels', 'los', 'shipman', 'activity',
        'supremacist', 'brother', 'armed', 'pistol', 'grasso', 'jury', 'baseline', 'white',
        'mysterious', 'youthful', 'naming', 'caseload',
        'brazen', 'uninsurance', 'reconviction', 'response', 'tampering', 'knife', 'bali', 'aces',
        'botched', 'woode', 'buy', 'sandwich',
        'solido', 'tools', 'harold', 'facility', 'mentally', 'nonviolent', 'lewd', 'ace', 'plead',
        'car', 'parole', 'break', 'petit',
        'offense', 'safe', 'marc', 'pantry', 'similar', 'pornography', 'carlton', 'classic',
        'richard', 'johnny', 'justice', 'stole',
        'spiking', 'spear', 'forcible', 'juvenile', 'knowledge', 'home', 'imprison', 'jigsaw',
        'pink', 'repeat', 'mischief', 'bungle',
        'norinco', 'sam', 'petty', 'siders', 'expert', 'squad', 'scenery', 'joe', 'artist',
        'prolific', 'chew', 'refuse', 'conceal',
        'habitual', 'ins', 'trespass', 'queen', 'advanced', 'reincarceration', 'distraction',
        'postponed', 'misconduct', 'calabrian',
        'cop', 'prisoner', 'ram', 'imperial', 'infection', 'blue', 'jeremy', 'black', 'computer',
        'handbag', 'detroit', 'chucky',
        'brandish', 'ice', 'incarceration', 'security', 'identity', 'mai', 'antoni', 'prisons',
        'prosecution', 'misappropriation',
        'scare', 'cobra', 'mexican', 'teen', 'notorious', 'tin', 'levi', 'walter', 'sider', 'v',
        'crime', 'masked', 'dowler', 'aryan',
        'misdemeanor', 'torrio', 'emergency', 'handcuff', 'detachment', 'human', 'debs', 'mickey',
        'mandatory', 'defilement', 'instrument',
        'agency', 'hilltop', 'drug', 'deportation', 'zodiac', 'efrain', 'cold', 'hobo', 'naples',
        'spousal', 'uttering', 'dui', 'targeted',
        'attempt', 'detain', 'exploitation', 'incarcerated', 'unregistere', 'neighborhood',
        'reconvict', 'moor', 'mortality', 'harbor',
        'meditate', 'testify', 'scared', 'hiv', 'ee', 'sheets', 'disposition', 'joanna', 'refused',
        'virus', 'wedlock', 'ex', 'caller',
        'reentry', 'statutory', 'lords', 'exposure', 'deputy', 'convict', 'confined', 'money',
        'metal', 'trojan', 'operation', 'reduce',
        'angel', 'collar', 'hospitalization', 'apt', 'rate', 'bust', 'threat', 'straight',
        'custody', 'liberty', 'card', 'dr', 'vito',
        'purse', 'sicilian', 'guilty', 'doll', 'jail', 'serial', 'fatal', 'revocation', 'alleged',
        're', 'spike', 'return', 'recognizance',
        'variant', 'deuces', 'accused', 'capias', 'nonforcible', 'do', 'nathan', 'pregnancy',
        'conviction', 'malicious', 'lover',
        'substance', 'misdemeanant', 'psychiatric', 'persistent', 'melbourne', 'welfare',
        'syndicate', 'detainment', 'pietro', 'graffiti',
        'crimes', 'winkler', 'raoul', 'infringement', 'infant', 'doc', 'own', 'victim', 'ar',
        'street', 'krishna', 'bundy', 'calculate',
        'indecently', 'within', 'rested', 'forcibly', 'lynn', 'net', 'harden', 'ill', 'worm',
        'neapolitan', 'path', 'carnal', 'lab',
        'langner', 'daytime', 'whereabout', 'tony', 'bamber', 'clear', 'odometer', 'criminal',
        'arresting', 'bandit', 'birney', 'reducing',
        'bandana', 'rascal', 'lonely', 'bredel', 'ira', 'digital', 'tire', 'uniform', 'baron',
        'biker', 'oo', 'cleared', 'target', 'touching',
        'queens', 'indictment', 'pre', 'allege', 'slashing', 'destruction', 'patronize', 'child',
        'apprehend', 'crush', 'enzyme', 'chinese',
        'charge', 'thirst', 'offend', 'policing', 'moh', 'force', 'probation', 'overcrowd',
        'teodor', 'custodial', 'pect', 'prevalence',
        'dropout', 'teenage', 'arraignment', 'gay', 'shoot', 'yorkshire', 'bike', 'u', 'insane',
        'victimization', 'tiny', 'store', 'incident',
        'baby', 'minimum', 'antisocial', 'mass', 'family', 'project', 'offender', 'la', 'capia',
        'rearreste', 'waive', 'hardened', 'mad',
        'confine', 'converter', 'centers', 'gord', 'sting', 'domain', 'german', 'man', 'deuce',
        'deliveryman', 'detection', 'release', 'police',
        'catalytic', 'proceeding', 'dump', 'nostra', 't', 'surrender', 'apartment', 'center',
        'behind', 'meth', 'bullying', 'batterer', 'felon',
        'willful', 'birth', 'genovese', 'consensual', 'converters', 'wooded', 'overcrowded',
        'unregistered', 'reconvicted', 'zeus', 'knifepoint', 'apprehension', 'vigilante', 'brutally',
        'upstart', 'botch', 'skinhead', 'consentual', 'mongol', 'gadi', 'helburn', 'probationers', 'porn', 'vijay',
        'soprano', 'redeploy', 'hells', 'waives', 'brandished', 'viruses', 'worms', 'ipr', 'returned', 'poisoner',
        'arrests', 'darko', 'rearresting', 'stolen', 'recidivate', 'reoffending', 'jilt', 'handcuff', 'dundon', 'chews',
        'brandishing', 'patriarca', 'neighborhoods', 'txt', 'offenders', 'crossbow', 'saric', 'chicano', 'rostov', 'yob',
        'breakin', 'evron', 'schoolgirl', 'testifies', 'jails', 'jilted', 'dentention', 'minimums', 'incarcerating', 'gp',
        'bungled', 'jou', 'areste', 'statuatory', 'nets', 'kuan', 'enzymes', 'jailing', 'smokin', 'transnational',
        'compounder', 'msg', 'convenience', 'nellessen'
    }