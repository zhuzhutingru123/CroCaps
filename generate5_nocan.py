import json
import logging
import random
import time
import copy
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import copy
import numpy as np

import torch
import torch.nn.functional as F
from evaluation import *
from transformers import *
from VideoGPT2_rank import *
from train5_rank_test import SPECIAL_TOKENS, SPECIAL_TOKENS_DICT
from dataset5_rank import get_dataset_test, build_input_from_segments2,build_input_from_segments, build_input_from_segments1,get_sub_frames
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def greedy_decode(cand,caption, tokenizer, model, args, current_output=None, video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    ys = []
    video = video.unsqueeze(0)
    cand=[]
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments2([ys], cand,tokenizer, with_eos=False, drop_caption=False,train=False)


        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        input_embs = model.transformer.wte(input_ids)
        if video is not None:
            input_embs = torch.cat([model.video_ff(video.cuda()), input_embs], dim=1)
            token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)

        logits = model(input_embs, token_type_ids=token_type_ids)
        if "gpt2" == args.model:
            logits = logits[0][0]
        logits = logits.cpu().data.numpy()
        next_word = np.argsort(logits[-1])[-1]
        if next_word == special_tokens_ids[1]:
            break
        ys.append(next_word)
    return ys




def sample_sequence(caption, tokenizer, model, args, current_output=None, video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    ys = []
    video = video.unsqueeze(0)

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments1([ys], tokenizer, with_eos=False, drop_caption=False,train=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        input_embs = model.transformer.wte(input_ids)

        if video is not None:
            input_embs = torch.cat([model.video_ff(video.cuda()), input_embs], dim=1)
            token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)

        logits = model(input_embs, token_type_ids=token_type_ids)
        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        ys.append(prev.item() )
    return ys

def beam_search(caption, tokenizer, model, args, current_output=None, video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    hyplist = [([], 0., current_output)]
    best_state = None
    comp_hyplist = []
    
    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            instance, sequence = build_input_from_segments(caption, history, st, tokenizer, with_eos=False, drop_caption=True)

            input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
            input_embs = model.transformer.wte(input_ids)
            if video is not None:
                input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
                token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)

            logits = model(input_embs, token_type_ids=token_type_ids)
            if "gpt2" == args.model:
                logits = logits[0]
            logp = F.log_softmax(logits, dim=-1)[:, -1, :]
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if i >= args.min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + args.penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return maxhyps
    else:
        return [([], 0)]


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)
# Evaluation routine
def generate_response(tokenizer,model, features_rgb, features_flow,attibute_fea, dataset, args, ref_data=None):

    candidate1 = json.load(open(args.candidate_path, 'r'))
    candidate = {}
    for num in range(len(candidate1)):

        videoid = candidate1[num]['image_id']

        if videoid not in candidate:
            candidate[videoid] = []

        candiadate_caption = [tokenize(candidate1[num]['dialog'], tokenizer)]
        candidate[videoid].append(candiadate_caption)
    ########################
    result_dialogs = []
    all_decoded_for_eval={}
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for idx in range(len(dataset)):

            vid = dataset[idx]['vid']
            #vid_fea = 'vid' + str(int(vid[5:]) + 1)
            tag_feature = attibute_fea[int(vid[3:])-1]
            cand=candidate[vid][0]
            fff1 = open("/home/zyh/wanru/data/MSVD/Vid2Url_Full.txt", 'r+')
            dic1 = eval(fff1.read())
            fea = features_rgb[dic1[vid]].value
            fea2 = features_flow[dic1[vid]].value
            feature_2d = get_sub_frames(fea, 30)
            feature_flow1 = get_sub_frames(fea2, 30)
            fea1 = []


            caption =dataset[idx]['caption']

            fea1 = []
            for j in range(30):
                #fea1.append(list(np.hstack((fea[j],fea2[j]))))
                #fea1.append(list(np.hstack((tag_feature))))
                fea1.append(list(np.hstack((feature_2d[j], feature_flow1[j], tag_feature))))

            fea1 = torch.Tensor(fea1).float()
            #fea1 = torch.zeros(fea1.shape[0], fea1.shape[1]).cuda()

            #hypstr = sample_sequence(caption, tokenizer, model, args, video=fea1)
            hypstr = greedy_decode(cand,caption, tokenizer, model, args, video=fea1)

            hypstr = tokenizer.decode(hypstr, skip_special_tokens=True)

            logging.info('vid: ' + vid)
            logging.info('HYP: ' + hypstr)

            pred_dialog = {'image_id': vid,
                           'dialog': hypstr}
            result_dialogs.append(pred_dialog)
            logging.info('ElapsedTime: %f' % (time.time() - start_time))



            if vid not in all_decoded_for_eval:
                all_decoded_for_eval[vid] = []

            all_decoded_for_eval[vid].append(hypstr)



            logging.info('-----------------------')


    return result_dialogs,all_decoded_for_eval


##################################
# main
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    #parser.add_argument("--model_checkpoint", type=str, default="log_1_dropout0101_ionput03_zeros/",help="Path, url or short name of the model")

    parser.add_argument("--model_checkpoint", type=str, default="log_1_dropout03_a/", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", action='store_true', help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=10, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float, default=0.3, help="elngth penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

    parser.add_argument("--test_path", type=str, default="/home/zyh/wanru/data/MSVD/msvd_sents_test_lc_nopunc.txt",
                        help="Path of the validset")
    parser.add_argument("--fea_path_test", type=str, default="/home/zyh/wanru/data/MSVD/5frame_test.txt",
                        help="Path of the trainset")

    parser.add_argument("--fea_path_rgb", type=str,
                        default="/home/zyh/wanru/data/MSVD/MSVD_InceptionV4.hdf5", help="Path of the trainset")
    parser.add_argument("--fea_path_flow", type=str,
                        default="/home/zyh/wanru/data/MSVD/MSVD_3DResNext101.hdf5", help="Path of the trainset")

    parser.add_argument("--namemap1", type=str,
                        default="/home/zyh/wanru/data/MSVD/Vid2Url_Full.txt", help="Path of the trainset")

    parser.add_argument("--att_path", type=str, default="msvd_semantic_tag_e1000.npy", help="Log path")

    parser.add_argument("--lbl_test_set", type=str, default="data/lbl_undisclosedonly_test_set4DSTC7-AVSD.json")
    parser.add_argument("--output", type=str, default="result.json")
    parser.add_argument("--candidate_path", type=str, default="result_test.json", help="Path, url or short name of the model")


    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    attibute_fea = np.load(args.att_path)
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + args.model_checkpoint)
    
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model_class = VideoGPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model_config = GPT2Config.from_pretrained(args.model_checkpoint)
    f = open('./loss_19_nocan.txt', 'w')
    model = model_class.from_pretrained(args.model_checkpoint+"checkpoint_mymodel_19.pth", config=model_config)
    model.to(args.device)
    model.eval()



    logging.info('Loading test data from ' )

    test_dataset, features_test,test_captions = get_dataset_test(tokenizer, args.test_path, args.fea_path_test, n_history=args.max_history)
    features_rgb = h5py.File(args.fea_path_rgb, 'r')
    features_flow = h5py.File(args.fea_path_flow, 'r')

    ref_decoded = {}
    for num in range(len(test_captions)):

        videoid = test_captions[num][0]

        if videoid not in ref_decoded:
            ref_decoded[videoid] = []

        ref_decoded[videoid].append(test_captions[num][1])



    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    result,all_decoded_for_eval = generate_response(tokenizer,model, features_rgb, features_flow,attibute_fea,test_dataset, args)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')





    scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)


    #f.write('Epoch %d\n' % epoch)
    f.write('\n')
    f.write("Bleu_1:" + str(scores['Bleu_1']))
    f.write('\n')
    f.write("Bleu_2:" + str(scores['Bleu_2']))
    f.write('\n')
    f.write("Bleu_3:" + str(scores['Bleu_3']))
    f.write('\n')
    f.write("Bleu_4:" + str(scores['Bleu_4']))
    f.write('\n')
    f.write("ROUGE_L:" + str(scores['ROUGE_L']))
    f.write('\n')
    f.write("METEOR:" + str(scores['METEOR']))
    f.write('\n')
    f.write("CIDEr:" + str(scores['CIDEr']))
    f.write('\n')
    f.close()
