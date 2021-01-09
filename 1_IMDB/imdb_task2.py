from mrjob.job import MRJob
from mrjob.step import MRStep
import re

import spacy
from spacy_langdetect import LanguageDetector
import fr_core_news_sm
import en_core_web_sm

# Regular expression to match words
WORD_RE = re.compile(r"[\w']+")

# Load language processing related models
NLP = spacy.load('en_core_web_sm')
NLP_FR = fr_core_news_sm.load()
NLP_EN = en_core_web_sm.load()
NLP.add_pipe(LanguageDetector(), name='language_detector', last=True)


class MostCommonKeyWordsByGenreIMDB(MRJob):

    def mapper_title_by_genre(self, _, line):
        """
        This mapper filters the line by the title type (second column)
        and maps each primary title (third column) to its genre (ninth column)
        :param _: None
        :param line: one line from the input file
        :return: (genre, primary_title)
        """

        # For input file type TSV, split line by tabs
        attributes = line.split('\t')

        # Columns: tconst titleType primaryTitle originalTitle isAdult startYear endYear runtimeMinutes genres
        title_type = attributes[1]
        primary_title = attributes[2]
        genre_list = attributes[8]

        if title_type == 'short':
            genres = genre_list.split(',')
            for genre in genres:
                yield genre.lower(), primary_title

    def mapper_keywords_by_genre(self, genre, title):
        """
        This mapper filters the words in the title by part of speech and
        yields each (genre, keyword) pair.
        :param genre: None
        :param title: primary title of movie
        :return: ((genre, keyword), 1)
        """

        # Detect the language of the title to correctly annotate the parts of speech
        doc = NLP(title)
        lang = doc._.language.get("language")
        if lang == 'en':
            doc = NLP_EN(title)
        else:
            doc = NLP_FR(title)

        for w in doc:
            if w.pos_ not in ('ADP', 'AUX', 'CONJ', 'CCONJ', 'DET', 'PUNCT', 'SCONJ', 'SYM', 'PART', 'X'):
                yield (genre, w.text.lower()), 1

    def combiner_count_words(self, genre_keyword_pair, counts):
        """
        This combiner sums the words we've selected so far by key
        :param genre_keyword_pair: (genre, keyword)
        :param counts: 1
        :return: ((genre, keyword), sum)
        """
        yield (genre_keyword_pair, sum(counts))

    def reducer_count_words(self, genre_keyword_pair, counts):
        """
        This reducer sends all (num_occurrences, (genre, keyword)) constructs to the next step
        :param genre_keyword_pair: pair obtained from the combiner
        :param counts: the number of occurrences of the key from the result of the combiner
        :return: (None, (sum(counts), genre_keyword_pair))
        """
        yield None, (sum(counts), genre_keyword_pair)

    def mapper_keyword_counts_by_genre(self, _, counts_genre_keyword_pair):
        """
        This mapper remaps the result of the previous step for our final reducer
        :param _: None
        :param counts_genre_keyword_pair: (count, (genre, keyword))
        :return: (genre, (count, keyword))
        """
        count = counts_genre_keyword_pair[0]
        genre = counts_genre_keyword_pair[1][0]
        keyword = counts_genre_keyword_pair[1][1]
        yield genre, (count, keyword)

    def reducer_find_top_fifteen_words_by_genre(self, genre, word_count_pairs):
        """
        This reducer gets the top 15 most common keywords for each movie genre
        :param genre: movie genre
        :param word_count_pairs: each item of word_count_pairs is (count, word)
        :return: (genre, (key=counts, value=word)) 15 times per genre
        """

        # Sort by keyword occurrence
        sorted_word_count_pairs = sorted(word_count_pairs, key=lambda x: x[0], reverse=True)
        top_range = len(sorted_word_count_pairs)
        if top_range >= 15:
            top_range = 15
        for i in range(top_range):
            yield genre, sorted_word_count_pairs[i]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_title_by_genre),
            MRStep(mapper=self.mapper_keywords_by_genre,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(mapper=self.mapper_keyword_counts_by_genre,
                   reducer=self.reducer_find_top_fifteen_words_by_genre)
        ]


if __name__ == '__main__':
    MostCommonKeyWordsByGenreIMDB.run()
