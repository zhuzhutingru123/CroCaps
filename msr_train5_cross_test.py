# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import json
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import *
from msvd_VideoGPT2_cross import *
import pickle as pkl
import h5py
from evaluation import *
from msvd_dataset5_cross import get_dataset_msvd,get_dataset_msr,get_dataset_msvd_best,get_dataset_msr_best,get_dataset, AVSDDataSet,  collate_fn,build_input_from_segments1,build_input_from_segments2,get_dataset_test
import time

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

logger = logging.getLogger(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_data_loaders_new(par,args, tokenizer,attibute_fea_msvd,attibute_fea_msr):
    train_data, features_train,ref_decoded= get_dataset_msvd_best(par,args,tokenizer, args.train_path_msvd, args.fea_path_train_msvd,  n_history=args.max_history)

    train_data1, features_train1, ref_decoded1 = get_dataset_msr_best(par, args, tokenizer, args.train_path_msr,args.fea_path_train_msr, n_history=args.max_history)

    valid_data, features_valid ,ref_decoded= get_dataset_msvd_best(par,args, tokenizer, args.train_path_msvd, args.fea_path_train_msvd,
                                                          n_history=args.max_history)

    #valid_data, features_valid ,ref_decoded= get_dataset(args,tokenizer, args.valid_path, args.fea_path_valid,  n_history=args.max_history)

    train_dataset = AVSDDataSet(train_data, tokenizer, attibute_fea_msvd,attibute_fea_msr,features_train, drop_rate=0, train=True)
    valid_dataset = AVSDDataSet(valid_data, tokenizer, attibute_fea_msvd,attibute_fea_msr,features_valid,  drop_rate=0, train=False)
    train_dataset1 = AVSDDataSet(train_data1, tokenizer, attibute_fea_msvd,attibute_fea_msr, features_train1, drop_rate=0, train=True)

    #train_dataset.dialogs.extend(train_dataset.dialogs)
    #train_dataset.dialogs.extend(train_dataset.dialogs)
    train_dataset.dialogs.extend(train_dataset1.dialogs)


    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=(not args.distributed), collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    valid_loader = DataLoader(valid_dataset, batch_size=1200*8, num_workers=4, shuffle=False,
                              collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))

    #valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    return train_loader,ref_decoded,valid_loader




def greedy_decode(cand, tokenizer, model, args,  video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    ys = []

    video = video.unsqueeze(0)
    cand=[]
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments2([ys], [cand],tokenizer, with_eos=False, drop_caption=False,train=False)



        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.cat([torch.ones(
            (1, 1 + 30)).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]),
                                    token_type_ids], dim=1)

        input_embs = model.transformer.wte(input_ids)




        if video is not None:
            video_embs_share = model.video_ff_share(video.cuda())
            #video_embs_private = model.video_ff_private(video.cuda())
            video_embs_share_mean = video_embs_share.mean(dim=1)
            #video_embs_private_mean = video_embs_private.mean(dim=1)
            video_embs_share_mean = video_embs_share_mean.reshape(1, 1, 768)
            #video_embs_private_mean = video_embs_private_mean.reshape(1, 1, 768)

            #input_embs = torch.cat([model.video_ff(video.cuda()), input_embs], dim=1)
            input_embs = torch.cat(
                [ video_embs_share_mean, video_embs_share, input_embs],
                dim=1)

            #token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)


        logits = model(input_embs, input_embs2=input_embs,token_type_ids1=token_type_ids,token_type_ids2=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0][0]

        logits = logits.cpu().data.numpy()

        next_word = np.argsort(logits[-1])[-1]

        if next_word == special_tokens_ids[1]:
            break
        ys.append(next_word)

    return ys


def generate_response_greedy(tokenizer, model, candidate,fea,vid, args):

    all_decoded_for_eval = {}
    model.eval()
    lable_predict=[]
    with torch.no_grad():

        for i in range(args.train_batch_size):
            fea1 = fea[i]
            cand=candidate[vid[i]][0]
            #hypstr = sample_sequence(caption, tokenizer, model, args, video=fea1)
            #fea1 = torch.Tensor(fea1).float()

            hypstr = greedy_decode(cand, tokenizer, model, args, video=fea1)
            hypstr.append(50259)
            lable_predict.append(hypstr)

            hypstr = tokenizer.decode(hypstr, skip_special_tokens=True)

            if vid[i] not in all_decoded_for_eval:
                all_decoded_for_eval[vid[i]] = []

            all_decoded_for_eval[vid[i]].append(hypstr)

    return all_decoded_for_eval,lable_predict

def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

def get_sub_frames(frames, K):
	# from all frames, take K of them, then add end of video frame
	if len(frames) < K:

		temp_zeros = np.zeros([K-frames.shape[0], frames.shape[1]])
		frames_ = np.concatenate((frames,temp_zeros), axis=0)
	else:
		index = np.linspace(0,len(frames),K,endpoint=False,dtype=int)
		frames_ = frames[index]
	return frames_

def generate_response_msr(tokenizer,model, features_rgb, features_flow,attibute_fea, dataset, args, ref_data=None):


    ########################
    result_dialogs = []
    all_decoded_for_eval={}
    all_decoded_for_eval1 = {}
    model.eval()
    with torch.no_grad():

        for idx in range(len(dataset)):

            vid = dataset[idx]['vid']

            tag_feature = attibute_fea[int(vid[5:])]
            cand=[]
            vid_fea = 'vid' + str(int(vid[5:]) + 1)


            fea = features_rgb[vid_fea].value
            fea2 = features_flow[vid].value



            feature_2d = get_sub_frames(fea, 30)
            feature_flow1 = get_sub_frames(fea2, 30)
            fea1 = []
            for j in range(30):
                #fea1.append(list(np.hstack((tag_feature))))
                fea1.append(list(np.hstack((feature_2d[j], feature_flow1[j], tag_feature))))

            fea1 = torch.Tensor(fea1).float()
            #fea1 = torch.zeros(fea1.shape[0], fea1.shape[1]).cuda()



            hypstr1 = greedy_decode(cand, tokenizer, model, args, video=fea1)
            hypstr1 = tokenizer.decode(hypstr1, skip_special_tokens=True)


            logging.info('vid: ' + vid)
            logging.info('HYP: ' + hypstr1)

            pred_dialog = {'image_id': vid,
                           'dialog': hypstr1}
            result_dialogs.append(pred_dialog)




            if vid not in all_decoded_for_eval:
                all_decoded_for_eval[vid] = []

            all_decoded_for_eval[vid].append(hypstr1)





            logging.info('-----------------------')


    return result_dialogs,all_decoded_for_eval


def generate_response(tokenizer,model, features_rgb, features_flow,attibute_fea, dataset, args, ref_data=None):


    ########################
    result_dialogs = []
    all_decoded_for_eval={}
    all_decoded_for_eval1 = {}
    model.eval()
    with torch.no_grad():

        for idx in range(len(dataset)):

            vid = dataset[idx]['vid']

            tag_feature = attibute_fea[int(vid[3:])-1]
            cand=[]
            fff1 = open("/home/zyh/wanru/data/MSVD/Vid2Url_Full.txt", 'r+')
            dic1 = eval(fff1.read())

            fea = features_rgb[dic1[vid]].value
            fea2 = features_flow[dic1[vid]].value



            feature_2d = get_sub_frames(fea, 30)
            feature_flow1 = get_sub_frames(fea2, 30)
            fea1 = []
            for j in range(30):
                #fea1.append(list(np.hstack((tag_feature))))
                fea1.append(list(np.hstack((feature_2d[j], feature_flow1[j], tag_feature))))

            fea1 = torch.Tensor(fea1).float()
            #fea1 = torch.zeros(fea1.shape[0], fea1.shape[1]).cuda()



            hypstr1 = greedy_decode(cand, tokenizer, model, args, video=fea1)
            hypstr1 = tokenizer.decode(hypstr1, skip_special_tokens=True)


            logging.info('vid: ' + vid)
            logging.info('HYP: ' + hypstr1)

            pred_dialog = {'image_id': vid,
                           'dialog': hypstr1}
            result_dialogs.append(pred_dialog)




            if vid not in all_decoded_for_eval:
                all_decoded_for_eval[vid] = []

            all_decoded_for_eval[vid].append(hypstr1)





            logging.info('-----------------------')


    return result_dialogs,all_decoded_for_eval

def train():
    parser = ArgumentParser()

    parser.add_argument("--train_path_msvd", type=str,
                        default="/home/zyh/wanru/data/MSVD/msvd_sents_train_noval_lc_nopunc.txt",
                        help="Path of the trainset")
    parser.add_argument("--fea_path_train_msvd", type=str, default="/home/zyh/wanru/data/MSVD/5frame_train.txt",
                        help="Path of the trainset")

    parser.add_argument("--train_path_msr", type=str, default="/home/zyh/wanru/data/MSRVTT/sent_train_file",
                        help="Path of the trainset")
    parser.add_argument("--fea_path_train_msr", type=str, default="/home/zyh/wanru/data/MSRVTT/5frame_train.txt",
                        help="Path of the trainset")

    parser.add_argument("--fea_path_rgb_msvd", type=str,
                        default="/home/zyh/wanru/data/MSVD/MSVD_InceptionV4.hdf5", help="Path of the trainset")
    parser.add_argument("--fea_path_flow_msvd", type=str,
                        default="/home/zyh/wanru/data/MSVD/MSVD_3DResNext101.hdf5", help="Path of the trainset")

    parser.add_argument("--fea_path_rgb_msr", type=str,
                        default="/home/zyh/wanru/data/MSRVTT/msrvtt_inpRes_rgb/feats.hdf5", help="Path of the trainset")
    parser.add_argument("--fea_path_flow_msr", type=str,
                        default="/home/zyh/wanru/data/MSRVTT/dataset_msrvtt/3DResNext101_train.hdf5",
                        help="Path of the trainset")
    parser.add_argument("--fea_path_flow_msr_test", type=str,
                        default="/home/zyh/wanru/data/MSRVTT/dataset_msrvtt/3DResNext101_test.hdf5",
                        help="Path of the trainset")

    parser.add_argument("--valid_path", type=str, default="/home/zyh/wanru/data/MSVD/msvd_sents_test_lc_nopunc.txt",
                        help="Path of the validset")
    parser.add_argument("--fea_path_valid", type=str, default="/home/zyh/wanru/data/MSVD/5frame_test.txt",
                        help="Path of the trainset")
    '''
    parser.add_argument("--test_path", type=str, default="/home/zyh/wanru/data/MSVD/msvd_sents_test_lc_nopunc.txt",
                        help="Path of the validset")
    parser.add_argument("--fea_path_test", type=str, default="/home/zyh/wanru/data/MSVD/5frame_test.txt",
                        help="Path of the trainset")
    '''
    parser.add_argument("--test_path", type=str, default="/home/zyh/wanru/data/MSRVTT/sent_test_file",
                        help="Path of the validset")
    parser.add_argument("--fea_path_test", type=str, default="/home/zyh/wanru/data/MSRVTT/5frame_test.txt",
                        help="Path of the trainset")



    parser.add_argument("--namemap1", type=str,
                        default="/home/zyh/wanru/data/MSVD/Vid2Url_Full.txt", help="Path of the trainset")




    #parser.add_argument("--fea_path_train", type=str, default="/home/zyh/wanru/data/MSRVTT/msrvtt_inpRes_rgb/feats.hdf5", help="Path of the trainset")

    #parser.add_argument("--fea_path_valid", type=str, default="/home/zyh/wanru/data/MSRVTT/msrvtt_inpRes_rgb/feats.hdf5", help="Path of the trainset")
    parser.add_argument("--max_length", type=int, default=10, help="Maximum length of the output utterances")

    #parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="100_step1234/", help="Path, url or short name of the model")
    parser.add_argument("--log_path", type=str, default="100_step1234/", help="Log path")

    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")#default=4
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop rate for caption")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")#default=6.25e-5,

    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")#default=8
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs")  # default=8
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    parser.add_argument("--train_cr_path_msr", type=str, default="train_best_cr_msr.json", help="Log path")
    parser.add_argument("--train_cr_path_msvd", type=str, default="train_best_cr_msvd.json", help="Log path")

    parser.add_argument("--att_path_msvd", type=str, default="msvd_semantic_tag_e1000.npy", help="Log path")
    parser.add_argument("--att_path_msr", type=str, default="msrvtt_e800_tag_feats.npy", help="Log path")
    args = parser.parse_args()
    attibute_fea_msvd = np.load(args.att_path_msvd)
    attibute_fea_msr = np.load(args.att_path_msr)
    par=100



    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')



    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")

    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model_class = VideoGPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model_config = GPT2Config.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint + "checkpoint_mymodel_19.pth", config=model_config)
    '''
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)8
    model_class = VideoGPT2LMHeadModel

    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    model_config = GPT2Config.from_pretrained(args.model_checkpoint)
    model_config.attn_pdrop = 0.3
    model_config.embd_pdrop = 0.3
    model_config.resid_pdrop = 0.3
    model_config.summary_first_dropout = 0.3
    model_config.n_layer = 2
    lamada = 1


    model = model_class.from_pretrained(args.model_checkpoint, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    '''





    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    #optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)



    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last) no entrance this two if
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")

    train_loader,ref_decoded1,valid_loader = get_data_loaders_new(par,args, tokenizer,attibute_fea_msvd,attibute_fea_msr)
    test_dataset, features_test, test_captions = get_dataset_test(tokenizer, args.test_path, args.fea_path_test,
                                                                  n_history=args.max_history)
    #features_rgb = h5py.File(args.fea_path_rgb_msvd, 'r')
    #features_flow = h5py.File(args.fea_path_flow_msvd, 'r')
    features_rgb = h5py.File(args.fea_path_rgb_msr, 'r')
    features_flow = h5py.File(args.fea_path_flow_msr_test, 'r')

    ref_decoded = {}
    for num in range(len(test_captions)):

        videoid = test_captions[num][0]

        if videoid not in ref_decoded:
            ref_decoded[videoid] = []

        ref_decoded[videoid].append(test_captions[num][1])




    def update(engine, batch):
        model.train()
        #batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids,token_type_ids,labels, input_mask, i3d, video_mask, reply_mask,score,vid ,score1,domain= batch


        x = torch.ones([i3d.size(0), 1], dtype=torch.int64) * (-1)
        labels = torch.cat([x, labels], dim=1)
        label_domain_neg = torch.ones([i3d.size(0), labels.size(1)], dtype=torch.int64) * (-1)

        label_domain_pos = torch.ones([i3d.size(0), 31], dtype=torch.int64) * (-1)


        #domain=domain.tolist()
        for i in range(i3d.size(0)):
            label_domain_neg[i,0]=domain[i]
            label_domain_pos[i, 0] = domain[i]

            if domain[i]==0:
                labels[i,]=torch.ones([1, labels.size(1)],dtype=torch.int64) * (-1)


        x = torch.ones([i3d.size(0), 31],dtype=torch.float32)
        #reply_mask = torch.cat([x, reply_mask], dim=1)
        #video_mask = torch.cat([x, video_mask], dim=1)
        reply_mask = torch.cat([x, reply_mask[:,30:]], dim=1)
        video_mask = torch.ones([reply_mask.size(0), reply_mask.size(1)],dtype=torch.float32)
        video_mask2 = torch.ones([reply_mask.size(0), 31], dtype=torch.float32)


        x = torch.ones([i3d.size(0), 1],dtype=torch.uint8)

        input_mask = torch.cat([x,  input_mask], dim=1)
        input_mask2 = torch.cat([x, input_mask[:, 0:30]], dim=1)

        input_ids = input_ids.to(args.device)# [8,14] juzi
        token_type_ids = token_type_ids.to(args.device) #[8,14] 50259 houmian all 50262
        i3d = i3d.to(args.device)  # [8,30,3884]

        labels = labels.to(args.device) #[8,44] qian 30 -1 hou 14juzi
        input_mask = input_mask.to(args.device) # all 1 [8,44]
        video_mask = video_mask.to(args.device) # qian 30 0,hou 14 1 [8,44]

        input_mask2 = input_mask2.to(args.device)  # all 1 [8,44]
        video_mask2 = video_mask2.to(args.device)  # qian 30 0,hou 14 1 [8,44]
        reply_mask = reply_mask.to(args.device) # all 0 [8,44]
        #input_mask_reply = input_mask_reply.to(args.device)
        label_domain_pos = label_domain_pos.to(args.device)
        label_domain_neg = label_domain_neg.to(args.device)







        input_embs1 = model.transformer.wte(input_ids)

        #greedy_caption,lable_predict = generate_response_greedy(tokenizer, model, candidate,i3d,vid, args)
        #reward = evaluate_captions_cider1(ref_decoded1, greedy_caption)



        video_embs_share = model.video_ff_share(i3d)
        video_embs_private = model.video_ff_private(i3d)
        video_embs_share_mean=video_embs_share.mean(dim=1)
        video_embs_private_mean = video_embs_private.mean(dim=1)
        video_embs_share_mean=video_embs_share_mean.reshape(i3d.size(0),1,768)
        video_embs_private_mean=video_embs_private_mean.reshape(i3d.size(0), 1, 768)

        i3d_mean=i3d.mean(dim=1)
        i3d_mean=i3d_mean.reshape(i3d.size(0),1,3884)

        label_i3d=torch.cat([i3d_mean,i3d], dim=1)


        #video_embs=torch.zeros(video_embs.shape[0],video_embs.shape[1],video_embs.shape[2]).cuda()
        #input_embs = torch.cat([video_embs, input_embs], dim=1)
        input_embs1 = torch.cat([video_embs_share_mean,video_embs_share,input_embs1], dim=1)
        input_embs2 = torch.cat([video_embs_private_mean, video_embs_private], dim=1)

        token_type_ids1 = torch.cat([torch.ones((i3d.size(0), 1+i3d.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)
        token_type_ids2 =torch.ones((i3d.size(0), 1 + i3d.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2])


        # [8,44] qian 30 50261 50259 houmian 50262
        video_loss = model(input_embs1,input_embs2=input_embs2,
                           token_type_ids1=token_type_ids1, token_type_ids2=token_type_ids2,
                           labels1=(labels, label_i3d), labels2=(labels, label_i3d),
                           attention_mask1=[video_mask, input_mask],attention_mask2=[video_mask2, input_mask2], mode="video")[0]

        reply_loss = model(input_embs1, input_embs2=input_embs2, token_type_ids1=token_type_ids1, token_type_ids2=token_type_ids2,labels1=(labels, label_i3d),
                           attention_mask1=[reply_mask, input_mask], attention_mask2=[video_mask2, input_mask2],mode="reply")[0]



        video_dif_loss = model(input_embs1,input_embs2=input_embs2,
                           token_type_ids1=token_type_ids1, token_type_ids2=token_type_ids2,
                           labels1=(labels, label_i3d), labels2=(labels, label_i3d),
                           attention_mask1=[video_mask, input_mask],attention_mask2=[video_mask2, input_mask2],  mode="video_diff")[0]

        video_domain_loss = model(input_embs1,input_embs2=input_embs2,
                           token_type_ids1=token_type_ids1, token_type_ids2=token_type_ids2,
                           labels1=(labels, label_domain_neg), labels2=(labels, label_domain_pos),
                           attention_mask1=[video_mask, input_mask],attention_mask2=[video_mask2, input_mask2],  mode="domain")[0]

        #reply_loss = model(input_embs,token_type_ids=token_type_ids, labels=(labels, label_i3d), attention_mask=[reply_mask,input_mask_reply], mode="reply")[0]
        #rank_loss = model(input_embs,lable_predict=lable_predict, reward=reward,score=score,bestscore=score1,token_type_ids=token_type_ids, labels=(labels, i3d), attention_mask=[reply_mask, input_mask],mode="rank")[0]

        #loss3 = rank_loss/ args.gradient_accumulation_steps
        loss1 = video_loss / args.gradient_accumulation_steps
        loss2 = reply_loss / args.gradient_accumulation_steps
        loss3 = video_dif_loss / args.gradient_accumulation_steps
        loss4= video_domain_loss / args.gradient_accumulation_steps


        #loss=min(loss1,loss2)
        loss = loss1+loss2+loss3+loss4
        #loss=loss1

        #logger.warning("loss %f %f", loss,loss1)
        #loss = rank_loss/3+ reply_loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm, norm_type=2)


        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)


    def inference(engine, batch):
        logger.info("inference datasets")
        f = open(args.log_path + 'result.txt', 'a')

        #result, all_decoded_for_eval1 = generate_response(tokenizer, model, features_rgb, features_flow, attibute_fea_msvd,test_dataset, args)
        result, all_decoded_for_eval1 = generate_response_msr(tokenizer, model, features_rgb, features_flow,attibute_fea_msr,test_dataset, args)

        scores = evaluate_for_particular_captions(all_decoded_for_eval1, ref_decoded)
        f.write("n_epochs:" + str(args.epochs))
        args.epochs = args.epochs+1
        f.write('\n')
        f.write("Bleu_4:" + str(scores['Bleu_4']))
        f.write('\n')
        f.write("ROUGE_L:" + str(scores['ROUGE_L']))
        f.write('\n')
        f.write("CIDEr:" + str(scores['CIDEr']))
        f.write('\n')
        #f.write("METEOR:" + str(scores['METEOR']))
        f.write('\n')
        f.write('\n')

        f.close()



        return 0

    evaluator = Engine(inference)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
    '''
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(valid_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(valid_loader))
    '''

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    #for name, metric in metrics.items():
        #metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        #evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir="./tb_logs")
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        #tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', save_interval=1, n_saved=20,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, args.log_path + 'model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))
        tokenizer.save_vocabulary(args.log_path)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.log_path, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()




if __name__ == "__main__":
    train()
