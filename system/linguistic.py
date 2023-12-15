from sklearn.feature_extraction.text import CountVectorizer
import pandas
import stanza
import argparse
from statistics import mean
from tqdm import tqdm

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file (JSON)")
    parser.add_argument("-o", "--output", type=str, help="Output file (csv)")
    return parser.parse_args()

def num_capitalized(text):
  tokens = text.split()
  return sum([1 for tok in tokens if tok[0].isupper()]) / len(tokens)

def count_punct(text):
    puncts = ".,?!:;-()"
    punct_count = 0
    for char in text:
        if char in puncts:
            punct_count += 1
    return punct_count / len(text)

def linguistic_features(text):
    dic = {}
    dic["num_capitalized"] = num_capitalized(text)
    dic["punct_count"] = count_punct(text)
    return dic

def main():
  args = create_arg_parser()
  with pandas.read_json(args.input, lines=True, chunksize=100) as chunks:
      for chunk in tqdm(chunks):
          results = chunk.apply(lambda x: linguistic_features(x['text']), axis=1, result_type="expand")
          results.loc[:, results.columns!='text'].to_csv(f"batch/batch_{chunk.index[0]}.csv")

main()