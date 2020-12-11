# Word Sense Disambiguation: HyperLex

In this assignment, the basic task is to take a raw corpus (which will be uploaded here), and run the hyperlex algorithm on it. HyperLex, as the class already knows, is an unsupervised WSD algorithm, where the aim is to take all the possible collocations of a word.

The algorithm of HyperLex is as follows:

1. Tokenize the corpus into sentences, and words. Clean the corpus by removing stopwords (using NLTK)
2. Define a collocation in a neighborhood. Create pairs of words that occur in the same sentence three words before and three words after. For example, in the sentence: "These people topical notions daily" -> ('topical', 'notions'), ('topical', 'daily') etc.
3. Rank these collocations based on frequency and create a graph (which words occur with which words the most). The words in the corpus would be the nodes, their cooccurrence defines an edge, and the frequency of the cooccurrence is the weight of that edge
4. Choose any single focus word, and for all the words in its direct neighbourhood and for words which are two-hops away, remove the most popular nodes, until the graph centered around that focus word splits into subgraphs. 

These words are those which show up in these clusters are those which are most associated with those senses of the word. And voila! Your very own HyperLex implementation. Your assignment is to implement the algorithm described above such that we can input any word in Step 4.

Link to paper: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.6499&rep=rep1&type=pdf