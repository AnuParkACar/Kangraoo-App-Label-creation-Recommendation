import spacy

nlp = spacy.load('en_core_web_sm')

text = "The Delaware PD is looking to hire an Apple employee to add to its force. Any Microsoft employees should be based in seattle. Ex Google employees are not welcome"

doc = nlp(text)

locations = [sentence for sentence in doc.sents if any(
    ent.label_ == "GPE" for ent in sentence.ents)]
print(locations)
