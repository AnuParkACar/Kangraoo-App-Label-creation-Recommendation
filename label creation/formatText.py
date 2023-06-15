class FormatText:
    def __init__(self, text: str) -> None:
        self.text = text

    def detect_whitespace(self, start: int, sentence: str) -> int:
        sentence = sentence[start:]
        for i, element in enumerate(sentence):
            if not element.isspace():
                return i

    def get_text(self) -> list:
        for i, entry in enumerate(self.text):
            entry_index_len = int(len(str(i)))
            entry_start_index = self.detect_whitespace(entry_index_len, entry)
            self.text[i] = entry[entry_start_index + entry_index_len:]
            print(self.text[i])
        return self.text
