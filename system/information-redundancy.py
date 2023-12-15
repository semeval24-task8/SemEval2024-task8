import stanza
from sklearn.feature_extraction.text import CountVectorizer
import torch
import pandas
import argparse
from tqdm import tqdm

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file (JSON)")
    parser.add_argument("-o", "--output", type=str, help="Output file (csv)")
    return parser.parse_args()

language_to_modelname = {
	"arabic": "ar",
	"russian": "ru",
	"chinese": "zh-hans",
	"indonesian": "id",
	"urdu": "ur",
	"bulgaria": "bg",
	"german": "de",
	"english": "en"
}

loaded_models = {}

def load_model(lang):
	if not lang in loaded_models:
		modelname = language_to_modelname.get(lang, "en")
		model = stanza.Pipeline(lang=modelname, processors="tokenize")
		loaded_models[lang] = model
		return model
	else:
		return loaded_models[lang]

def get_model(lang):
	if not lang in language_to_modelname:
		return loaded_models.get("english", load_model("english"))
	else:
		return loaded_models.get(lang, load_model(lang))

def sentences(text, lang):
	nlp = get_model(lang)
	doc = nlp(text)
	return [s.text.strip() for s in doc.sentences]

def sent_term_matrix(sentences):
	vect = CountVectorizer()
	mat = vect.fit_transform(sentences)
	return torch.from_numpy(mat.toarray()).float()

def sent_sent_matrix(m):
	return torch.mm(torch.transpose(m, 0, 1), m)

def truncate(m, k):
	u, s, v = torch.svd_lowrank(m, q=min(6, *m.shape[-2:]))
	n = round(k * s.shape[0])
	s[-n:] = 0
	return u.mm(torch.diag_embed(s)).mm(v.transpose(0, 1))

def info_loss_metrics(orig, trunc):
	diff = torch.sub(orig, trunc)
	norm = torch.linalg.matrix_norm(diff).pow(2).item()
	trunc_max = torch.max(trunc).item()
	trunc_min = torch.min(trunc).item()
	trunc_mean = torch.mean(trunc).item()
	trunc_median = torch.median(trunc).item()
	sum_diff = torch.sum(diff).item()
	return {"info_red_loss_norm": norm,
				 "info_red_trunc_max": trunc_max,
				 "info_red_trunc_min": trunc_min,
				 "info_red_trunc_mean": trunc_mean,
				 "info_red_trunc_median": trunc_median,
				 "info_red_sum_diff": sum_diff}

def information_redundancy(sample, lang):
	sample_sentences = sentences(sample, lang)
	term_matrix = sent_term_matrix(sample_sentences)
	sent_matrix = sent_sent_matrix(term_matrix)
	truncated = truncate(sent_matrix, 0.25)
	return info_loss_metrics(sent_matrix, truncated)

def main():
	args = create_arg_parser()
	data = pandas.read_json(args.input, lines=True)
	tqdm.pandas()
	data = data.join(data.progress_apply(lambda x: information_redundancy(x["text"], x["source"]), axis=1, result_type="expand"))
	data[["id", "label", "info_red_loss_norm",
			 "info_red_trunc_max", "info_red_trunc_min",
			 "info_red_trunc_mean", "info_red_trunc_median",
			 "info_red_sum_diff"]].to_csv(args.output, index=False)
	
main()