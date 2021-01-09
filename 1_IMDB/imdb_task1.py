from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import nltk
import fr_core_news_sm

NLP_FR = fr_core_news_sm.load()

nltk.download('stopwords')

# Regular expression to match words
WORD_RE = re.compile(r"[\w']+")

# Stopwords are used to determine the language
# The method to detect the English language using NLTK can be found here:
# https://www.algorithm.co.il/programming/python/cheap-language-detection-nltk/
ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS

STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}


def is_english(text):
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)


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

            if is_english(primary_title):
                for word in WORD_RE.findall(primary_title):

                    # Filter out auxiliary verbs, prepositions, articles and conjunctions
                    # Available parts of speech can be listed with nltk.help.upenn_tagset()
                    # List also available at the official documentation:
                    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
                    nltk_tagged_word = nltk.pos_tag([word])
                    part_of_speech = nltk_tagged_word[0][1]

                    if part_of_speech not in ('IN', 'RP', 'CC', 'MD', 'DT', 'PDT', 'TO'):
                        yield word.lower(), 1
            else:
                # The second most common language in the input file was French, so we decided to
                # filter out auxiliary verbs, prepositions, articles and conjunctions in French;
                # This is done using spaCy.io
                doc = NLP_FR(primary_title)
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
