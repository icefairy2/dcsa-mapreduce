from mrjob.job import MRJob
from mrjob.step import MRStep
import re

import spacy
from spacy_langdetect import LanguageDetector
import fr_core_news_sm
import en_core_web_sm

# Regular expression to match words
WORD_RE = re.compile(r"[\w']+")


class MostCommonKeyWordsIMDB(MRJob):

    def mapper_filter_by_title_type_and_part_of_speech(self, _, line):
        """
        This mapper filters the line by the title type (second column)
        and then filters each word in the primary title (third column)
        by part of speech.
        :param _: None
        :param line: one line from the input file
        :return: (word, 1)
        """

        # For input file type TSV, split line by tabs
        attributes = line.split('\t')

        # Columns: tconst titleType primaryTitle originalTitle isAdult startYear endYear runtimeMinutes genres
        title_type = attributes[1]
        primary_title = attributes[2]

        if title_type in ('short', 'movie'):
            nlp = spacy.load('en_core_web_sm')
            nlp_fr = fr_core_news_sm.load()
            nlp_eng = en_core_web_sm.load()
            nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

            doc = nlp(primary_title)
            lang = doc._.language.get("language")
            if lang == 'en':
                doc = nlp_eng(primary_title)
            else:
                doc = nlp_fr(primary_title)

            for w in doc:
                if w.pos_ not in ('ADP', 'AUX', 'CONJ', 'CCONJ', 'DET', 'PUNCT', 'SCONJ', 'SYM', 'PART', 'X'):
                    yield w.text.lower(), 1

    def combiner_count_words(self, word, counts):
        """
        This combiner sums the words we've selected so far
        :param word: word obtained from the mapper
        :param counts: 1
        :return: (word, sum)
        """
        yield (word, sum(counts))

    def reducer_count_words(self, word, counts):
        """
        This reducer sends all (num_occurrences, word) pairs to the same final reducer.
        :param word: word obtained from the combiner
        :param counts: the number of occurrences of the word from the result of the combiner
        :return: (None, (sum(counts), word))
        """
        yield None, (sum(counts), word)

    def reducer_find_top_fifty_words(self, _, word_count_pairs):
        """
        This reducer gets the top 50 most common keywords
        :param _: discard the key; it is just None
        :param word_count_pairs: each item of word_count_pairs is (count, word),
        :return: (key=counts, value=word) 50 times
        """

        # Sort by keyword occurrence
        sorted_word_count_pairs = sorted(word_count_pairs, key=lambda x: x[0], reverse=True)
        for i in range(50):
            yield sorted_word_count_pairs[i]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_filter_by_title_type_and_part_of_speech,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(reducer=self.reducer_find_top_fifty_words)
        ]


if __name__ == '__main__':
    MostCommonKeyWordsIMDB.run()
