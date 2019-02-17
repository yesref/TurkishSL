import jpype as jp


# noinspection PyMethodMayBeStatic
class TurkishMorphemeAnalyzer:
    def __init__(self, path_prefix):
        self._init_jvm(path_prefix + 'lib/zemberek-full.jar')
        self.morphology = jp.JClass('zemberek.morphology.TurkishMorphology').createWithDefaults()
        self.index_out_of_bounds_exception = jp.JClass('java.lang.IndexOutOfBoundsException')

    def _init_jvm(self, zemberek_path):
        if jp.isJVMStarted():
            return
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % zemberek_path)

    def _convert_list_to_sentence(self, word_list, conjunction):
        return conjunction.join(str(word) for word in word_list)

    def _get_token_analysis(self, word_analyses, index):
        try:
            token_analysis = word_analyses[index]
        except jp.JException(self.index_out_of_bounds_exception):
            return None, None

        best_analysis = token_analysis.getBestAnalysis()
        token = token_analysis.getWordAnalysis().getInput()

        return token, best_analysis

    def _get_next_word(self, word_list, index):
        try:
            return word_list[index]
        except jp.JException(self.index_out_of_bounds_exception):
            return None

    def get_sentence_morphemes(self, word_list):
        sentence = self._convert_list_to_sentence(word_list, ' ')
        stc_analysis = self.morphology.analyzeAndDisambiguate(sentence)
        word_analyses = stc_analysis.wordAnalyses

        token_index = 0
        morpheme_list = []
        for curr_word in word_list:
            curr_token, curr_token_analysis = self._get_token_analysis(word_analyses, token_index)
            word_morphemes = [morpheme.morpheme.id for morpheme in curr_token_analysis.morphemeDataList]

            token_length = len(curr_token)
            while token_length < len(curr_word):
                next_token, next_token_analysis = self._get_token_analysis(word_analyses, token_index + 1)
                if next_token is not None:
                    word_morphemes += [morpheme.morpheme.id for morpheme in next_token_analysis.morphemeDataList]
                    token_length += len(next_token)
                    token_index += 1
                else:
                    break

            morpheme_list.append(self._convert_list_to_sentence(word_morphemes, ','))
            token_index += 1

        if len(word_list) != len(morpheme_list):
            raise RuntimeError('Inappropriate Morpheme Detected for this sentence: %s' % sentence)

        return morpheme_list


def test(sentence):
    analyzer = TurkishMorphemeAnalyzer('../')
    # sentence_morphemes = [analyzer.get_sentence_morphemes(stc) for stc in sentences]
    sentence_morphemes = analyzer.get_sentence_morphemes(sentence)
    jp.shutdownJVM()
    return sentence_morphemes


if __name__ == "__main__":
    morphemes = test('Ankara \'da saat 15:30 \'da Dr. Mehmet bey ile görüşeceğiz'.split(' '))
    print(morphemes)
