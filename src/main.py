import logging
from typing import List

import numpy as np
import torch
from numpy import ndarray
from sklearn.cluster import KMeans
from spacy.lang.en import English
from transformers import *
import streamlit as st

logging.basicConfig(level=logging.WARNING)
import spacy

device = "cpu"
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

default_value_goes_here = """
New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. 
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. 
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. 
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. 
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 
2010 marriage license application, according to court documents. 
Prosecutors said the marriages were part of an immigration scam. 
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. 
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective 
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002. 
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. 
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages. 
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. 
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s 
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. 
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. 
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

class SentenceExtractor(object):
    def __init__(self, language=English):
        self.nlp = language()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        doc = self.nlp(body)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        return self.process(body, min_length, max_length)

def extract_embeddings(text, polling_type = 'mean'):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    token_tensor = torch.tensor([indexed_tokens]).to(device)
    hidden_states = model(token_tensor)[0]
    pooled = hidden_states.mean(dim=1)
    return pooled

def feature_matrix(sentences):
    return np.asarray([
            np.squeeze(extract_embeddings(t).data.cpu().numpy())
            for t in sentences
        ])

def find_closest_args(centroids, text_embeddings):
    centroid_min = 1e10
    cur_arg = -1
    args = {}
    used_idx = []
    for j, centroid in enumerate(centroids):
        for i, feature in enumerate(text_embeddings):
            value = np.linalg.norm(feature - centroid)
            if value < centroid_min and i not in used_idx:
                cur_arg = i
                centroid_min = value
        used_idx.append(cur_arg)
        args[j] = cur_arg
        centroid_min = 1e10
        cur_arg = -1
    return args

def cluster(text_embeddings, ratio= 0.5):
    k = 1 if ratio * len(text_embeddings) < 1 else int(len(text_embeddings) * ratio)
    model = KMeans(k).fit(text_embeddings)
    centroids = model.cluster_centers_
    cluster_args = find_closest_args(centroids, text_embeddings)
    sorted_values = sorted(cluster_args.values())
    return sorted_values

def generate_summary(sentences, text_embeddings, ratio_value):
    hidden_args = cluster(text_embeddings, ratio = ratio_value)
    if hidden_args[0] != 0:
        hidden_args.insert(0,0)
    return [sentences[j] for j in hidden_args]



###
### Main UI
###

@st.cache()
def load_bert(name = 'distilbert-base-cased'):
    model = DistilBertModel.from_pretrained(name)
    tokenizer = DistilBertTokenizer.from_pretrained(name)
    return model, tokenizer


@st.cache()
def load_all():
    model, tokenizer = load_bert()
    return model, tokenizer

model, tokenizer = load_all()


st.title("Summarizarion with DistillBERT clusters")
st.write('torch version  is', torch.__version__)


user_input = st.text_area("Text to Summarize", default_value_goes_here, height = 400)


sentence_extractor = SentenceExtractor()
sentences = sentence_extractor(user_input, 40, 400)
#st.write('Sentences: ', sentences)
summ_size = st.slider('Summary size', 1, 10, 1)
if st.button('Summarize'):
    text_embeddings = feature_matrix(sentences)
    summary = generate_summary(sentences, text_embeddings, ratio_value = summ_size*0.1)
    '''
    ### Summary
    '''
    summary = ' '.join(summary)
    st.write(HTML_WRAPPER.format(summary), unsafe_allow_html=True)
else:
    st.write('Click to summarize')

