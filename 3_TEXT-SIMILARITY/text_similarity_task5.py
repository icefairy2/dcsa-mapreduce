import json
import math

import nltk
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

from nltk.tokenize import sent_tokenize, word_tokenize

from gensim import corpora
from gensim import models

INPUT_PROTOCOL = JSONValueProtocol

RANDOM_PAPER = {"author": "[{'name': 'Ahmed Osman'}, {'name': 'Wojciech Samek'}]",
                "day": 1,
                "id": "1802.00209v1",
                "link": "[{'rel': 'alternate', 'href': 'http://arxiv.org/abs/1802.00209v1', 'type': 'text/html'}, {'rel': 'related', 'href': 'http://arxiv.org/pdf/1802.00209v1', 'type': 'application/pdf', 'title': 'pdf'}]",
                "month": 2,
                "summary": "We propose an architecture for VQA which utilizes recurrent layers to\ngenerate visual and textual attention. The memory characteristic of the\nproposed recurrent attention units offers a rich joint embedding of visual and\ntextual features and enables the model to reason relations between several\nparts of the image and question. Our single model outperforms the first place\nwinner on the VQA 1.0 dataset, performs within margin to the current\nstate-of-the-art ensemble model. We also experiment with replacing attention\nmechanisms in other state-of-the-art models with our implementation and show\nincreased accuracy. In both cases, our recurrent attention mechanism improves\nperformance in tasks requiring sequential or relational reasoning on the VQA\ndataset.",
                "tag": "[{'term': 'cs.AI', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'cs.CL', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'cs.CV', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'cs.NE', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'stat.ML', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}]",
                "title": "Dual Recurrent Attention Units for Visual Question Answering",
                "year": 2018}

random_result = {}

nltk.download('punkt')


def compute_random_paper_aspects():
    """
    This method computes the computationally relevant data on the given random paper
    :return: random_result: the text to vec representation of the random paper's summary
    """

    random_paper_summary = RANDOM_PAPER["summary"]
    random_paper_summary = random_paper_summary.replace("\n", " ")
    random_data = []
    # Iterate through each sentence in the random paper summary
    for i in sent_tokenize(random_paper_summary):
        temp = []

        # Tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        random_data.append(temp)
    # Compute the dictionary of words for the random paper
    random_paper_dictionary = corpora.Dictionary(random_data)

    # From the dictionary of words, form a new dictionary of words in
    # their alphabetical order and their index in this alphabetical order
    random_paper_vector = random_paper_dictionary.token2id

    # For each sentence, convert each word into a tuple with the
    # following structure: word_id, nb of times that a word appears in the sentence
    random_corpus = [random_paper_dictionary.doc2bow(sentence) for sentence in random_data]

    # Compute the TF-IDF model for the random paper to obtain the term frequency
    # in the summary with respect to the length of the summary
    random_tfidf = models.TfidfModel(random_corpus)

    # In random_result, return a dictionary containing the words in their alphabetical
    # order and the total TF-IDF score of each word that appears in the random paper
    random_vec = []
    for document in random_tfidf[random_corpus]:
        for word, score in document:
            random_vec.append((word, score))
    num_dict = {}
    for t in random_vec:
        if t[0] in num_dict:
            num_dict[t[0]] = num_dict[t[0]] + t[1]
        else:
            num_dict[t[0]] = t[1]
    global random_result
    for k, v in random_paper_vector.items():
        if v in num_dict:
            random_result[k] = num_dict.get(v)
    return random_result


class SimilarPaperRecommendations(MRJob):

    def mapper_paper_summary(self, _, line):
        """
        This mapper yields the paper id and the paper summary for each paper in the JSON
        :param _: None
        :param line: one line from the input file
        :return: (paper_id, paper_summary)
        """

        # For input file type JSON, get the id and the summary of all the papers
        paper_data = json.loads(line)
        paper_id = paper_data["id"]
        paper_summary = paper_data["summary"]

        yield paper_id, paper_summary

    def mapper_compute_cosine_similarity(self, paper_id, paper_summary):
        """
        This mapper computes the cosine similarity between a random paper (given)
        and each paper in the JSON and yields the paper id, paper summary,
        and the cosine similarity between this paper and the randomly given paper
        :param paper_id: unique id as defined in the input data
        :param paper_summary: corresponding paper summary
        :return: ((paper_id, paper_summary), cosine_similarity)
        """

        global random_result
        paper_summary = paper_summary.replace("\n", " ")

        paper_data = []
        # Iterate through each sentence in the paper summary
        for i in sent_tokenize(paper_summary):
            temp = []

            # Tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())

            paper_data.append(temp)

        # Compute the dictionary of words for each paper
        paper_dictionary = corpora.Dictionary(paper_data)

        # From the dictionary of words, form a new dictionary of words in
        # their alphabetical order and their index in this alphabetical order
        paper_vector = paper_dictionary.token2id

        # For each sentence, convert each word into a tuple with the
        # following structure: word_id, nb of times that a word appears in the sentence
        paper_corpus = [paper_dictionary.doc2bow(sentence) for sentence in paper_data]

        # Compute the TF-IDF model for the paper to obtain the term frequency
        # in the summary with respect to the length of the summary
        paper_tfidf = models.TfidfModel(paper_corpus)

        # In paper_result, return a dictionary containing the words in their alphabetical
        # order and the total score of each word that appears in each paper
        paper_vec = []
        for document in paper_tfidf[paper_corpus]:
            for word, score in document:
                paper_vec.append((word, score))
        num_dict = {}
        for t in paper_vec:
            if t[0] in num_dict:
                num_dict[t[0]] = num_dict[t[0]] + t[1]
            else:
                num_dict[t[0]] = t[1]

        paper_result = {}
        for k, v in paper_vector.items():
            if v in num_dict:
                paper_result[k] = num_dict.get(v)

        # Compute the cosine similarity between each paper and the randomly selected paper,
        # but avoid this computation in case the paper and the random paper are the same;
        # The cosine similarity implementation was done without using a library that computes it
        # automatically, and can be found here:
        # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        # The formula used in the cosine similarity computation will be presented in
        # the documentation
        if paper_result != random_result:

            # Select the terms that are present in the paper and the random paper
            intersection = set(random_result.keys()) & set(paper_result.keys())
            numerator = sum([random_result[x] * paper_result[x] for x in intersection])
            sum1 = sum([random_result[x] ** 2 for x in random_result.keys()])
            sum2 = sum([paper_result[x] ** 2 for x in paper_result.keys()])
            denominator = math.sqrt(sum1) * math.sqrt(sum2)
            if not denominator:
                cosine_similarity = 0.0
            else:
                cosine_similarity = round(float(numerator) / denominator, 3)

            yield None, (cosine_similarity, (paper_id, paper_summary))

    def reducer_find_highest_similarity(self, _, similarity_paper_pair):
        """
        This reducer orders the list of papers by their similarity compared with
        the randomly selected paper and returns the paper which is most similar
        to the randomly selected paper
        :param _: None
        :param similarity_paper_pair:(cosine_similarity, (paper_id, paper_summary))
        :return: sorted_similarity_paper_pairs[0]:
        the (cosine_similarity, (paper_id, paper_summary)) construct for the paper
        with the highest similarity compared to the randomly selected paper
        """

        # Sort by cosine similarity of the paper
        sorted_similarity_paper_pairs = sorted(similarity_paper_pair, key=lambda x: x[0], reverse=True)
        yield sorted_similarity_paper_pairs[0]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_paper_summary),
            MRStep(mapper=self.mapper_compute_cosine_similarity,
                   reducer=self.reducer_find_highest_similarity)
        ]


if __name__ == '__main__':
    compute_random_paper_aspects()
    SimilarPaperRecommendations.run()
