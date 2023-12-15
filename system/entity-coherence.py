from sklearn.feature_extraction.text import CountVectorizer
import pandas
import argparse
from tqdm import tqdm
import stanza

nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse,coref")

def create_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, help="Input file (JSON)")
  parser.add_argument("-o", "--output", type=str, help="Output file (csv)")
  return parser.parse_args()

def entity_coherence(text):
  doc = nlp(text)
  role_rank = {
    "nsubj": 3,
    "nsubjpass": 3,
    "csubj": 3,
    "csubjpass": 3,
    "agent": 3,
    "obj": 2,
    "dative": 2
  }
  role_map = {
    "nsubj": "S",
    "nsubjpass": "S",
    "csubj": "S",
    "csubjpass": "S",
    "agent": "S",
    "obj": "O",
    "dative": "O",
    "-": "-"
  }
  roles = ["S","O","X","-"]

  chains = doc.coref
  mentions_by_sentence = [[[m.start_word for m in c.mentions if m.sentence == s] for s in range(len(doc.sentences))] for c in chains]
  roles_by_sentence = [[max([doc.sentences[i].words[m].deprel for m in s], key=lambda role: role_rank.get(role, 1)) if len(s) > 0 else "-" for i, s in enumerate(c)] for c in mentions_by_sentence]
  role_seq_by_sentence = ["".join([role_map.get(s, "X") for s in c]) for c in roles_by_sentence]

  possible_transitions = [a + b for a in roles for b in roles]
  vect = CountVectorizer(vocabulary=possible_transitions, ngram_range=(2,2), analyzer="char", lowercase=False)
  mat = vect.fit_transform(role_seq_by_sentence)
  feature_ids = vect.vocabulary_
  dist = mat.sum(axis=0) / mat.sum()
  features = {feature: dist[0, id] for feature, id in feature_ids.items()}
  return features

def main():
  args = create_arg_parser() 
  data = pandas.read_json(args.input, lines=True)
  tqdm.pandas()
  data = data.join(data.progress_apply(lambda x: entity_coherence(x["text"]), axis=1, result_type="expand"))

# main()

print(entity_coherence("I like Pontus. He is a good friend and he knows everything. I am friendly"))