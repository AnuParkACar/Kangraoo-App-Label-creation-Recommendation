import spacy
nlp = spacy.load("en_core_web_sm")


class Tags:
    def __init__(self, text) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = self.nlp(text)

    def get_tokens(self):
        for token in self.doc:
            print(token.text)
