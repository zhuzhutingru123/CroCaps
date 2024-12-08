import pickle as pickle
import os
import sys
sys.path.append('./coco-caption')

from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu

from collections import defaultdict

def score_all(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def score(ref, hypo):
    scorers = [
        #(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #(Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    #scorers = [
        #(Cider(), "CIDEr")
    #]
    #scorers = [
    #    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
    #]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores



def evaluate_for_particular_captions(cand, ref_captions):
    ref = ref_captions
    # with open(candidate_path, 'rb') as f:
    #     cand = pickle.load(f)

    # make dictionary
    hypo = {}
    refe = {}
    for key, caption in cand.items():
        hypo[key] = cand[key]
        refe[key] = ref[key]
    # compute bleu score
    final_scores = score_all(refe, hypo)

    # print out scores

    return final_scores
'''
def evaluate_captions_cider(ref, cand):
    #hypo = []

    #refe = defaultdict()
    #for i, caption in enumerate(cand):
    #    temp = defaultdict()
    #    temp['image_id'] = i
    #    temp['caption'] = [caption]
    #    hypo.append(temp)
    #    refe[i] = ref[i]
    #final_scores = score(refe, hypo)
    # # return final_scores['Bleu_1']
    # #### normal scores ###
    hypo = {}
    final_scores = defaultdict()
    refe = {}
    for i, caption in enumerate(cand):
         hypo[i] = [caption]
         refe[i] = ref[i]
    #score1, scores = Bleu(4).compute_score(refe, hypo)
    #method = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    #for m, s in zip(method, scores):
    #     final_scores[m] = s
    score1, scores = Rouge().compute_score(refe, hypo)
    final_scores['ROUGE_L'] = scores
    #
    # return 2 * final_scores['CiderD'] + 1 * final_scores['Bleu_4'] + 1*final_scores['ROUGE_L']
    return final_scores['ROUGE_L']
'''
def evaluate_captions_cider(ref, cand):
    #ref = [ref]
    #cand = [cand]

    hypo = {}
    refe = {}
    score3=[]


    for i, caption in enumerate(cand):
        #hypo[i] = [caption]
        #refe[i] = ref[i]
        refe[0] = ref[caption]
        hypo[0] = cand[caption]
        final_scores = score(refe, hypo)
        score3.append(final_scores['ROUGE_L'])

    # return final_scores['Bleu_1']
    return score3

def evaluate_captions_cider1(ref, cand):
    #ref = [ref]
    #cand = [cand]

    hypo = {}
    refe = {}



    for i, caption in enumerate(cand):
        #hypo[i] = [caption]
        #refe[i] = ref[i]
        refe[i] = ref[caption]
        hypo[i] = cand[caption]

    final_scores = score(refe, hypo)

    # return final_scores['Bleu_1']

    return final_scores['ROUGE_L']
    #return final_scores['CIDEr']+final_scores['ROUGE_L']
    #return final_scores['CIDEr']


def evaluate_captions_wrapper(args):
   return evaluate_captions(*args)

def evaluate_captions(ref, cand):
    ref = [ref]
    cand = [cand]
    hypo = {}

    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    final_scores = score(refe, hypo)
    # return final_scores['Bleu_1']
    #return 1 * final_scores['CIDEr']
    return final_scores['METEOR']


    #return final_scores['METEOR']

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" % (split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" % (split, split))

    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)

    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]

    # compute bleu score
    final_scores = score_all(ref, hypo)

    # print out scores
    print ('Bleu_1:\t', final_scores['Bleu_1'])
    print ('Bleu_2:\t', final_scores['Bleu_2'])
    print ('Bleu_3:\t', final_scores['Bleu_3'])
    print ('Bleu_4:\t', final_scores['Bleu_4'])
    print ('METEOR:\t', final_scores['METEOR'])
    print ('ROUGE_L:', final_scores['ROUGE_L'])
    print ('CIDEr:\t', final_scores['CIDEr'])

    if get_scores:
        return final_scores
