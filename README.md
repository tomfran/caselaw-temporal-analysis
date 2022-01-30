# Harvard Caselaw Access Project temporal analysis

The goal of the project is to apply information retrieval techniques to legal text, in particular, LDA topic modelling and Word Embeddings. 

The main idea is to study words in a temporal axis, finding trends regarding context or single or group of words, trends in frequency and regarding topics. 

The dataset in use is the [Illinois portion](https://case.law/bulk/download/) of the Harvard Caselaw Access Project.

## Methodology

The overall process can be divided in preprocessing, topic modelling and word embeddings.

A complete overview of the methodology and the results can be found on the [project report](https://github.com/tomfran/caselaw-temporal-analysis/blob/main/report/report.pdf).

### Preprocessing

This phase uses Spacy to tokenize the text and obtain a lemmatized version of each token.

### Topic modelling

The next phase involves finding topics in the dataset, optimizing the number of components using an Halving search approach. After finding a general overview of the data, a more refined topic modelling is run on a subset of the found topics.

### Word embeddings

The main idea of this part is to train Word2vec models on year and epochs of the data, a similar word can be found [here](https://github.com/williamleif/histwords), in fact, we give credits to them for the model alignment that makes all the analysis in this part possible.

## Webapp

The project is accessible through a [webapp](https://illinois-cases-analysis-webapp-qka7d4ktba-ew.a.run.app/), please mind the loading time required if an instance is not running, about 1 to 3 minutes.

It is possible to search for single or group of words, each query is separated by a minus, while a group is concatenated with the comma, e.g. cocaine, cannabis - gun, searches for two queries, the combination of cocaine and cannabis, and gun respectively. 

Here are some screenshots for the semantic shift part that exploit the words embeddings, and the topic modelling section.

![Semantic shift](https://github.com/tomfran/caselaw-temporal-analysis/blob/main/report/images/semantic_1.png?raw=true "Title")
![Semantic shift](https://github.com/tomfran/caselaw-temporal-analysis/blob/main/report/images/semantic_2.png?raw=true "Title")
![Topic modelling](https://github.com/tomfran/caselaw-temporal-analysis/blob/main/report/images/topic1.png?raw=true "Title")
![Topic modelling](https://github.com/tomfran/caselaw-temporal-analysis/blob/main/report/images/topic2.png?raw=true "Title")
![Topic modelling](https://github.com/tomfran/caselaw-temporal-analysis/blob/main/report/images/topic3.png?raw=true "Title")

