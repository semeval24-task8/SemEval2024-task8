from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import argparse
from tqdm import tqdm
import numpy

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file (JSON)")
    parser.add_argument("-o", "--output", type=str, help="Output file (csv)")
    return parser.parse_args()

def main():
  args = create_arg_parser()
  vect = TfidfVectorizer(ngram_range=(1,1), analyzer=("word"), max_df=0.8, min_df=0.2)
  tokenizer = vect.build_analyzer()
  vocab = set()
  with pandas.read_json(args.input, lines=True, chunksize=100) as chunks:
      for chunk in tqdm(chunks):
          chunk_texts = list(chunk['text'])
          for text in chunk_texts:
              vocab.update(tokenizer(text))
  vect = TfidfVectorizer(ngram_range=(1,1), analyzer=("word"), max_df=0.8, min_df=0.2, vocabulary=vocab)
  with pandas.read_json(args.input, lines=True, chunksize=100) as chunks:
      for chunk in tqdm(chunks):
          chunk_texts = list(chunk['text'])
          mat = vect.fit_transform(chunk_texts)
          numpy.savetxt(f"batch/batch_{chunk.index[0].csv}", mat, ",")

main()