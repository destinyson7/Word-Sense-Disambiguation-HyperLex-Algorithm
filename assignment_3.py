from nltk.corpus import stopwords
import numpy as np
import random
from collections import deque


class Kruskals:
    def __init__(self, num_vertices, adjacency_matrix):
        self.num_vertices = num_vertices
        self.adjacency_matrix = adjacency_matrix

        self.parent = [i for i in range(self.num_vertices)]
        self.unionSize = [1 for i in range(self.num_vertices)]

        self.mst_adj = [[] for _ in range(self.num_vertices)]
        self.edges = []

        # print(adjacency_matrix)

        for i in range(num_vertices):
            for j in range(num_vertices):
                self.edges.append((self.adjacency_matrix[i][j], i, j))

        self.MAX = 1e18

    def find(self, x):
        if x == self.parent[x]:
            return x

        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, u, v):
        a = self.find(u)
        b = self.find(v)

        if self.unionSize[a] > self.unionSize[b]:
            a, b = b, a

        self.parent[a] = b
        self.unionSize[b] += self.unionSize[a]

    def solve(self):

        edges_taken = 0
        MST = []

        self.edges = sorted(self.edges)

        for edge in self.edges:
            if edges_taken == self.num_vertices - 1:
                break

            if self.find(edge[1]) != self.find(edge[2]):
                self.union(edge[1], edge[2])
                self.mst_adj[edge[1]].append(edge[2])
                self.mst_adj[edge[2]].append(edge[1])
                MST.append((edge[1], edge[2], edge[0]))
                edges_taken += 1

        return MST, self.mst_adj


class BFS:
    def __init__(self, adj, root_hubs, root):
        self.adj = adj
        self.root_hubs = root_hubs
        self.root = root
        self.num_vertices = len(self.adj)
        self.visited = [False for _ in range(self.num_vertices)]
        self.visited[root] = True
        self.components = []
        self.cur_component = []

    def bfs(self):

        for root in self.adj[self.root]:
            q = deque()
            q.append(root)

            self.cur_component = []

            while q:

                top = q.popleft()
                self.visited[top] = True
                self.cur_component.append(top)

                for vertex in self.adj[top]:
                    if not self.visited[vertex]:
                        self.visited[vertex] = True
                        q.append(vertex)

            self.components.append(self.cur_component)

        return self.components


class Hyperlex:
    def __init__(self):
        self.sentences = []

        self.other_stop_words = [
            "?", "#", "@", "--", ";", "%", "$", ".", ",", "`", "/", "\\"
        ]

        self.other_stop_words_full = ["&", ":"]

        self.cooccurences = {}

        self.WINDOW_SIZE = 3

        self.frequencies = {}
        self.THRESHOLD_FREQUENCY = 2
        self.THRESHOLD_WEIGHT = 0.8
        self.NUMBER_OF_NEIGHBOURS = 3

        self.VERY_LARGE_VALUE = 1e15

        self.extract()
        self.remove_stop_words()
        self.collocation_window()
        self.get_frequencies()
        self.take_input()

    def extract(self):
        with open("./corpus.txt", "r") as f:
            for line in f:
                text = line.split("[")[1].split("]")[0].strip()
                sentence = []

                for word in text.split(")"):

                    if word:
                        tagged = word.split("(")[1]

                        cur_word = ""

                        if tagged.find('"') >= 0 or tagged.find("-NONE-") >= 0:
                            pass

                        else:
                            cur_word = tagged.split("'")[1]
                            tag = tagged.split("'")[3]

                            isStopWord = False

                            if cur_word == tag:
                                isStopWord = True

                            for stop in self.other_stop_words:
                                if cur_word.find(stop) >= 0:
                                    isStopWord = True
                                    break

                            for stop in self.other_stop_words_full:
                                if cur_word == stop:
                                    isStopWord = True
                                    break

                            if not isStopWord and not cur_word.isnumeric() and (cur_word[0] != "-" or cur_word[-1] != "-"):
                                sentence.append(cur_word.lower())

                if len(sentence) > 0:
                    # print(" ".join(sentence))
                    self.sentences.append(sentence)

        # print(self.sentences)
        # print(len(self.sentences))

    def remove_stop_words(self):
        stop_words = set(stopwords.words('english'))

        temp = []

        for sentence in self.sentences:
            filtered_sentence = [w for w in sentence if not w in stop_words]

            if len(filtered_sentence) > 1:
                temp.append(filtered_sentence)

        self.sentences = temp

        # print(self.sentences)
        # print(len(self.sentences))

    def collocation_window(self):
        for sentence in self.sentences:
            length = len(sentence)

            for index in range(length):

                if sentence[index] in self.cooccurences.keys():
                    freq = self.cooccurences[sentence[index]]

                else:
                    freq = {}

                for j in range(max(0, index - self.WINDOW_SIZE), index):

                    if sentence[j] in freq.keys():
                        freq[sentence[j]] += 1

                    else:
                        freq[sentence[j]] = 1

                for j in range(index + 1, min(index + self.WINDOW_SIZE + 1, length)):

                    if sentence[j] in freq.keys():
                        freq[sentence[j]] += 1

                    else:
                        freq[sentence[j]] = 1

                self.cooccurences[sentence[index]] = freq

        # print(self.cooccurences["futures"])
        # print(len(self.cooccurences["futures"]))

    def get_frequencies(self):
        for sentence in self.sentences:
            for word in sentence:

                if word in self.frequencies.keys():
                    self.frequencies[word] += 1

                else:
                    self.frequencies[word] = 1

        # print(self.frequencies)

    def get_sorted_list_of_coocurrences(self, word):

        words_with_frequencies = sorted(
            self.cooccurences[word].items(), reverse=True, key=lambda x: x[1])

        # print(words_with_frequencies)

        return list(list(zip(*words_with_frequencies))[0])

    def construct_adjacency_matrix(self, word_coocurrences, word_at_index):
        adj = np.ones((len(word_coocurrences), len(word_coocurrences)))

        for a in range(len(word_coocurrences)):
            for b in range(len(word_coocurrences)):
                ff = word_at_index[a]
                ss = word_at_index[b]

                if ff in self.cooccurences.keys() and ss in self.cooccurences[ff].keys():
                    adj[a][b] = max(
                        0, min(adj[a][b], 1 - self.cooccurences[ff][ss] / self.frequencies[ss]))

                if ss in self.cooccurences.keys() and ff in self.cooccurences[ss].keys():
                    adj[a][b] = max(
                        0, min(adj[a][b], 1 - self.cooccurences[ss][ff] / self.frequencies[ff]))

        return adj

    def map_index_to_word(self, word_coocurrences):
        word_at_index = {}
        index_of_word = {}

        for i in range(len(word_coocurrences)):
            word_at_index[i] = word_coocurrences[i]
            index_of_word[word_coocurrences[i]] = i

        return word_at_index, index_of_word

    def remove_neightbours(self, G, V, v, index_of_word):
        new_V = []
        for word in V:
            # print(word, v)
            if G[index_of_word[v]][index_of_word[word]] < 1:
                continue

            else:
                new_V.append(word)

        return new_V

    def GoodCandidate(self, G, v, index_of_word):

        weights = G[index_of_word[v]]
        weights = list(filter(lambda x: x < 1, weights))

        if len(weights) < self.NUMBER_OF_NEIGHBOURS:
            return False

        weights.sort()
        mean_of_most_frequent_neighbors = sum(
            weights[:self.NUMBER_OF_NEIGHBOURS]) / self.NUMBER_OF_NEIGHBOURS

        return mean_of_most_frequent_neighbors < self.THRESHOLD_WEIGHT

    def RootHubs(self, G, V, target, index_of_word):
        H = []
        index_H = []

        # print(V)

        while len(V) > 0 and self.cooccurences[target][V[0]] > self.THRESHOLD_FREQUENCY:
            v = V.pop(0)

            if self.GoodCandidate(G, v, index_of_word):
                H.append(v)
                index_H.append(index_of_word[v])
                V = self.remove_neightbours(G, V, v, index_of_word)
                # print(V)

        return H, index_H

    def Components(self, G, H, t, index_of_word, word_at_index):

        for root in H:
            G[index_of_word[root]][index_of_word[t]] = -1
            G[index_of_word[t]][index_of_word[root]] = -1

        Kruskal = Kruskals(index_of_word[t] + 1, G)
        tree, tree_adj = Kruskal.solve()

        T = []

        for edge in tree:
            T.append((word_at_index[edge[0]], word_at_index[edge[1]]))

        return tree, T, tree_adj

    def take_input(self):

        word = input()
        word = word.lower()

        if word not in self.cooccurences.keys():
            print("Word is not in corpus after removing stop words")
            return

        word_coocurrences = self.get_sorted_list_of_coocurrences(word)
        # print(len(word_coocurrences))

        word_at_index, index_of_word = self.map_index_to_word(
            word_coocurrences)
        index_of_word[word] = len(word_coocurrences)
        word_at_index[len(word_coocurrences)] = word

        # print(index_of_word)

        adj = self.construct_adjacency_matrix(word_coocurrences, word_at_index)
        new_adj = np.pad(adj, ((0, 1), (0, 1)),
                         mode='constant',
                         constant_values=self.VERY_LARGE_VALUE)
        # print(np.shape(adj))
        # print(np.shape(new_adj))
        # print(new_adj)

        H, index_H = self.RootHubs(
            new_adj, word_coocurrences, word, index_of_word)

        if not H:
            print("Insuffiecient data in corpus to disambiguate this word")
            return

        # print(H)

        mst, T, mst_adj = self.Components(
            new_adj, H, word, index_of_word, word_at_index)
        # print(T)
        # print(mst)
        # print(mst_adj)

        BFSSolver = BFS(mst_adj, index_H, index_of_word[word])
        components = BFSSolver.bfs()

        # print(components)

        senses = []

        for component in components:
            words_in_sense = []

            for node in component:
                words_in_sense.append(word_at_index[node])

            senses.append(words_in_sense)

        # print(*senses, sep="\n")

        output = []

        for sense in senses:
            output.append(sense[:min(len(sense), 3)])

        answer = ""
        for sense in output:
            if len(sense) == 0:
                continue

            elif len(sense) == 1:
                answer += f"({sense[0]})" + ", "

            elif len(sense) == 2:
                answer += f"({sense[0]}, {sense[1]})" + ", "

            elif len(sense) == 3:
                answer += f"({sense[0]}, {sense[1]}, {sense[2]})" + ", "

        print(answer[:-2])


if __name__ == "__main__":
    Hyperlex()
