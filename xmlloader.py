# -*- coding: utf-8 -*-

import declxml as xml

class XmlLoader:
    def __init__(self):
        paraphrase_processor = xml.dictionary('paraphrase', [
            xml.array(xml.string('value'), alias='values')
        ])

        corpus_processor = xml.dictionary('corpus', [
            xml.array(paraphrase_processor, alias='paraphrases')
        ])

        self.data_processor = xml.dictionary('data', [
            xml.array(corpus_processor, alias='corpus')
        ])

    def load(self, path):
        file = open(path, 'r', encoding='UTF-8')
        self.file_contents = file.read()
        file.close()
        
    def parse(self):
        result = xml.parse_from_string(self.data_processor, self.file_contents)
        for item in result['corpus'][0]['paraphrases']:
            item = item['values']
            paraphraseObj = Paraphrase(item[0], item[1], item[2], item[3], item[4], item[6])
            yield paraphraseObj

class Paraphrase:
    def __init__(self, _id, id_1, id_2, string_1, string_2, value):
        self.id = int(_id)
        self.id_1 = int(id_1)
        self.id_2 = int(id_2)
        self.string_1 = string_1
        self.string_2 = string_2
        self.value = 0 if int(value) < 1 else 1