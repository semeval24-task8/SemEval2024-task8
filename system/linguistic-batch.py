from sklearn.feature_extraction.text import CountVectorizer
import pandas
import stanza
import argparse
from statistics import mean
from tqdm import tqdm
from stanza_batch import batch

# stanza.download(lang='en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

pos_vocab = ["ADJ", "ADP", "ADV",
             "AUX", "CCONJ", "DET",
             "INTJ", "NOUN", "NUM",
             "PART", "PRON", "PROPN",
             "PUNCT", "SCONJ", "SYM",
             "VERB", "X"]

vect = CountVectorizer(vocabulary=pos_vocab, lowercase=False)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file (JSON)")
    parser.add_argument("-o", "--output", type=str, help="Output file (csv)")
    parser.add_argument("-bs", "--batch_size", type=str, help="Batch size")
    parser.add_argument("-f", "--from_index", type=int, help="From index")
    parser.add_argument("-t", "--to_index", type=int, help="To index")
    return parser.parse_args()

def stanza_pos(doc):
    pos = [word.upos for sent in doc.sentences for word in sent.words]
    mat = vect.fit_transform([' '.join(pos)])
    freq = mat / mat.sum()
    feature_ids = vect.vocabulary_
    features = {feature: freq[0, id] for feature, id in feature_ids.items()}
    return features

def sent_length(doc):
    lengths = [len(s.tokens) for s in doc.sentences]
    return mean(lengths)

def batch_process(samples, batch_size):
    if batch_size == 1:
        return [nlp[text] for text in samples]
    else:
        return batch(samples, nlp, batch_size)

def linguistic_features(samples, batch_size):
    processed = list(batch_process(samples, batch_size))
    dic = {}
    dic["sent_length"] = [sent_length(doc) for doc in processed]
    pos = [stanza_pos(doc) for doc in processed]
    pos = {k: [x[k] for x in pos] for k, v in pos[0].items()}
    dic = dict(dic, **pos)
    return dic

def main():
  args = create_arg_parser()
  with pandas.read_json(args.input, lines=True, chunksize=10) as reader:
      for chunk in tqdm(reader):
          if chunk.index[0] >= args.from_index and chunk.index[-1] <= args.to_index:
              texts = list(chunk['text'])
              results = linguistic_features(texts, args.batch_size)
              pandas.DataFrame(results).to_csv(f"batch/ling_batch_{chunk.index[0]}.csv")

main()