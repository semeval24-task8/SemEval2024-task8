import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas
import argparse
from tqdm import tqdm
import coreferee

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

doc = nlp("I like Pontus. He is a good friend.")
x = doc.doc._.coref_chains[0][0][0]
print([i for i, s in enumerate(doc.sents) if x in range(s.start, s.end)][0])

def create_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, help="Input file (JSON)")
  parser.add_argument("-o", "--output", type=str, help="Output file (csv)")
  return parser.parse_args()

def entity_coherence(text):
  doc = nlp(text)
  role_mapping = {
    "nsubj": "S",
    "nsubjpass": "S",
    "csubj": "S",
    "csubjpass": "S",
    "agent": "S",
    "dobj": "O",
    "pobj": "O",
    "dative": "O",
    "_other": "X",
    "_none": "-"
  }
  roles = list(role_mapping.values())

  # TODO
  refs = doc._.coref_chains

  # TODO
  ref_roles = []

  ref_role_seq = [''.join(r) for r in ref_roles]
  possible_transitions = [a + b for a in roles for b in roles]
  vect = CountVectorizer(vocabulary=possible_transitions, ngram_range=(2,2), analyzer="char")
  mat = vect.fit_transform(ref_role_seq)
  feature_ids = vect.vocabulary_
  dist = mat.sum(axis=0) / mat.sum()
  features = {feature: dist[0, id] for feature, id in feature_ids.items()}
  return features

def main():
  args = create_arg_parser() 
  data = pandas.read_json(args.input, lines=True)
  tqdm.pandas()
  data = data.join(data.progress_apply(lambda x: entity_coherence(x["text"]), axis=1, result_type="expand"))

main()