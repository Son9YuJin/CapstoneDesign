# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
import csv
from argparse import Namespace
from pprint import pprint
from datasets import load_dataset 
from functools import partial
import pandas as pd

import numpy # for gradio hot reload
import gradio as gr

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def _normalize_detection_result(res, z_threshold=None):
    """
    detection 결과가 dict/tuple/list 어떤 형태로 와도
    {'is_detected','p_value','z_score'} 형태의 dict로 바꿔준다.
    """
    out = {'is_detected': 'N/A', 'p_value': 'N/A', 'z_score': 'N/A'}

    if isinstance(res, dict):
        out['p_value'] = res.get('p_value', res.get('p', 'N/A'))
        out['z_score'] = res.get('z_score', res.get('z', 'N/A'))
        out['is_detected'] = res.get('is_detected', res.get('detected', 'N/A'))
        return out

    if isinstance(res, (tuple, list)):
        # p_value 후보: 0~1 사이의 수
        for x in res:
            if isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0:
                out['p_value'] = float(x); break
        # z_score 후보: 실수들 중 0~1 제외한 첫 값
        nums = [float(x) for x in res if isinstance(x, (int, float))]
        z_alts = [z for z in nums if not (0.0 <= z <= 1.0)]
        if z_alts:
            out['z_score'] = z_alts[0]
        # is_detected 후보: bool 있으면 사용, 없으면 z_threshold로 판정
        bools = [b for b in res if isinstance(b, bool)]
        if bools:
            out['is_detected'] = bools[0]
        elif z_threshold is not None and isinstance(out['z_score'], (int, float)):
            out['is_detected'] = abs(float(out['z_score'])) >= float(z_threshold)
        return out

    return out


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        # 출력 토큰 수
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        # 프롬프트 토큰 수
        "--input_prompt_tokens",
        type=int,
        default=200,
        help="Number of tokens from the beginning of the prompt to use as model input.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None, is_watermarked=False):
    """
    텍스트를 생성하는 함수.
    워터마크 적용 여부(is_watermarked)에 따라 LogitsProcessor를 다르게 설정합니다.
    max_length가 비정상적으로 큰 값(예: 1e30 등)이 들어오는 경우를 방지하기 위해
    안전한 최대 길이를 계산하도록 수정했습니다.
    """

    # --- 워터마크 설정 ---
    if is_watermarked:
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme,
            select_green_tokens=args.select_green_tokens
        )
        logits_processor = LogitsProcessorList([watermark_processor])
    else:
        logits_processor = None

    # --- 생성 파라미터 설정 ---
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False
    )

    # input_ids에서 앞 200 토큰만 사용
    input_ids = enc["input_ids"][0][:200]
    # 길이에 맞는 attention mask (앞에서부터 연속 토큰이므로 전부 1)
    attention_mask = torch.ones_like(input_ids)

    # 배치 차원 추가 + 디바이스 이동
    tokd_input = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device)
    }

    # 시드 고정(옵션)
    torch.manual_seed(args.generation_seed)

    # --- 생성 ---
    output = model.generate(**tokd_input, logits_processor=logits_processor, **gen_kwargs)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return decoded_output



def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args

def detect_raw(input_text, args, device=None, tokenizer=None):
    """
    CSV 저장/로직 처리를 위한 '점수 dict'를 반환하는 감지 함수.
    UI용 detect()는 (표 렌더 데이터, args) 튜플을 반환하므로,
    코드 로직에서는 이 함수를 써서 p_value, z_score 등을 안정적으로 가져오세요.
    """
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens
    )

    if len(input_text) - 1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)  # <-- dict
        return score_dict  # keys: num_tokens_scored, num_green_tokens, green_fraction, z_score, p_value, prediction, confidence ...
    else:
        # 너무 짧으면 빈 dict
        return {}
    
def generate_ui(prompt, args, model=None, device=None, tokenizer=None):
    # UI에서는 재디코딩/트렁케이션 경고를 간단히 처리
    redecoded_input = prompt
    truncation_warning = 0  # 또는 False

    # 기본/워터마크 출력 각각 생성
    out_no_wm = generate(
        prompt, args, model=model, device=device, tokenizer=tokenizer, is_watermarked=False
    )
    out_wm = generate(
        prompt, args, model=model, device=device, tokenizer=tokenizer, is_watermarked=True
    )

    return redecoded_input, truncation_warning, out_no_wm, out_wm, args


def run_gradio(args, model=None, device=None, tokenizer=None):
    """Define and launch the gradio demo interface"""
    generate_ui_partial = partial(generate_ui, model=model, device=device, tokenizer=tokenizer)
    detect_partial = partial(detect, device=device, tokenizer=tokenizer)

    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                """
                ## 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍
                """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                """
                [![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jwkirchenbauer/lm-watermarking)
                """
                )
            # with gr.Column(scale=2):
            #     pass
            # ![visitor badge](https://visitor-badge.glitch.me/badge?page_id=tomg-group-umd_lm-watermarking) # buggy

        with gr.Accordion("Understanding the output metrics",open=False):
            gr.Markdown(
            """
            - `z-score threshold` : The cuttoff for the hypothesis test
            - `Tokens Counted (T)` : The number of tokens in the output that were counted by the detection algorithm. 
                The first token is ommitted in the simple, single token seeding scheme since there is no way to generate
                a greenlist for it as it has no prefix token(s). Under the "Ignore Bigram Repeats" detection algorithm, 
                described in the bottom panel, this can be much less than the total number of tokens generated if there is a lot of repetition.
            - `# Tokens in Greenlist` : The number of tokens that were observed to fall in their respective greenlist
            - `Fraction of T in Greenlist` : The `# Tokens in Greenlist` / `T`. This is expected to be approximately `gamma` for human/unwatermarked text.
            - `z-score` : The test statistic for the detection hypothesis test. If larger than the `z-score threshold` 
                we "reject the null hypothesis" that the text is human/unwatermarked, and conclude it is watermarked
            - `p value` : The likelihood of observing the computed `z-score` under the null hypothesis. This is the likelihood of 
                observing the `Fraction of T in Greenlist` given that the text was generated without knowledge of the watermark procedure/greenlists.
                If this is extremely _small_ we are confident that this many green tokens was not chosen by random chance.
            -  `prediction` : The outcome of the hypothesis test - whether the observed `z-score` was higher than the `z-score threshold`
            - `confidence` : If we reject the null hypothesis, and the `prediction` is "Watermarked", then we report 1-`p value` to represent 
                the confidence of the detection based on the unlikeliness of this `z-score` observation.
            """
            )

        with gr.Accordion("A note on model capability",open=True):
            gr.Markdown(
                """
                This demo uses open-source language models that fit on a single GPU. These models are less powerful than proprietary commercial tools like ChatGPT, Claude, or Bard. 

                Importantly, we use a language model that is designed to "complete" your prompt, and not a model this is fine-tuned to follow instructions. 
                For best results, prompt the model with a few sentences that form the beginning of a paragraph, and then allow it to "continue" your paragraph. 
                Some examples include the opening paragraph of a wikipedia article, or the first few sentences of a story. 
                Longer prompts that end mid-sentence will result in more fluent generations.
                """
                )
        gr.Markdown(f"Language model: {args.model_name_or_path} {'(float16 mode)' if args.load_fp16 else ''}")

        # Construct state for parameters, define updates and toggles
        default_prompt = args.__dict__.pop("default_prompt")
        session_args = gr.State(value=args)

        with gr.Tab("Generate and Detect"):
            
            with gr.Row():
                prompt = gr.Textbox(label=f"Prompt", interactive=True,lines=10,max_lines=10, value=default_prompt)
            with gr.Row():
                generate_btn = gr.Button("Generate")
            with gr.Row():
                with gr.Column(scale=2):
                    output_without_watermark = gr.Textbox(label="Output Without Watermark", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    # without_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    without_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
            with gr.Row():
                with gr.Column(scale=2):
                    output_with_watermark = gr.Textbox(label="Output With Watermark", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    # with_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"],interactive=False,row_count=7,col_count=2)

            redecoded_input = gr.Textbox(visible=False)
            truncation_warning = gr.Number(visible=False)
            def truncate_prompt(redecoded_input, truncation_warning, orig_prompt, args):
                if truncation_warning:
                    return redecoded_input + f"\n\n[Prompt was truncated before generation due to length...]", args
                else: 
                    return orig_prompt, args
        
        with gr.Tab("Detector Only"):
            with gr.Row():
                with gr.Column(scale=2):
                    detection_input = gr.Textbox(label="Text to Analyze", interactive=True,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    # detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
            with gr.Row():
                    detect_btn = gr.Button("Detect")

        # Parameter selection group
        with gr.Accordion("Advanced Settings",open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"#### Generation Parameters")
                    with gr.Row():
                        decoding = gr.Radio(label="Decoding Method",choices=["multinomial", "greedy"], value=("multinomial" if args.use_sampling else "greedy"))
                    with gr.Row():
                        sampling_temp = gr.Slider(label="Sampling Temperature", minimum=0.1, maximum=1.0, step=0.1, value=args.sampling_temp, visible=True)
                    with gr.Row():
                        generation_seed = gr.Number(label="Generation Seed",value=args.generation_seed, interactive=True)
                    with gr.Row():
                        n_beams = gr.Dropdown(label="Number of Beams",choices=list(range(1,11,1)), value=args.n_beams, visible=(not args.use_sampling))
                    with gr.Row():
                        max_new_tokens = gr.Slider(label="Max Generated Tokens", minimum=10, maximum=1000, step=10, value=args.max_new_tokens)

                with gr.Column(scale=1):
                    gr.Markdown(f"#### Watermark Parameters")
                    with gr.Row():
                        gamma = gr.Slider(label="gamma",minimum=0.1, maximum=0.9, step=0.05, value=args.gamma)
                    with gr.Row():
                        delta = gr.Slider(label="delta",minimum=0.0, maximum=10.0, step=0.1, value=args.delta)
                    gr.Markdown(f"#### Detector Parameters")
                    with gr.Row():
                        detection_z_threshold = gr.Slider(label="z-score threshold",minimum=0.0, maximum=10.0, step=0.1, value=args.detection_z_threshold)
                    with gr.Row():
                        ignore_repeated_bigrams = gr.Checkbox(label="Ignore Bigram Repeats")
                    with gr.Row():
                        normalizers = gr.CheckboxGroup(label="Normalizations", choices=["unicode", "homoglyphs", "truecase"], value=args.normalizers)
            # with gr.Accordion("Actual submitted parameters:",open=False):
            with gr.Row():
                gr.Markdown(f"_Note: sliders don't always update perfectly. Clicking on the bar or using the number window to the right can help. Window below shows the current settings._")
            with gr.Row():
                current_parameters = gr.Textbox(label="Current Parameters", value=args)
            with gr.Accordion("Legacy Settings",open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_separately = gr.Checkbox(label="Seed both generations separately", value=args.seed_separately)
                    with gr.Column(scale=1):
                        select_green_tokens = gr.Checkbox(label="Select 'greenlist' from partition", value=args.select_green_tokens)
        
        with gr.Accordion("Understanding the settings",open=False):
            gr.Markdown(
            """
            #### Generation Parameters:

            - Decoding Method : We can generate tokens from the model using either multinomial sampling or we can generate using greedy decoding.
            - Sampling Temperature : If using multinomial sampling we can set the temperature of the sampling distribution. 
                                0.0 is equivalent to greedy decoding, and 1.0 is the maximum amount of variability/entropy in the next token distribution.
                                0.7 strikes a nice balance between faithfulness to the model's estimate of top candidates while adding variety. Does not apply for greedy decoding.
            - Generation Seed : The integer to pass to the torch random number generator before running generation. Makes the multinomial sampling strategy
                                outputs reproducible. Does not apply for greedy decoding.
            - Number of Beams : When using greedy decoding, we can also set the number of beams to > 1 to enable beam search. 
                                This is not implemented/excluded from paper for multinomial sampling but may be added in future.
            - Max Generated Tokens : The `max_new_tokens` parameter passed to the generation method to stop the output at a certain number of new tokens. 
                                    Note that the model is free to generate fewer tokens depending on the prompt. 
                                    Implicitly this sets the maximum number of prompt tokens possible as the model's maximum input length minus `max_new_tokens`,
                                    and inputs will be truncated accordingly.
            
            #### Watermark Parameters:

            - gamma : The fraction of the vocabulary to be partitioned into the greenlist at each generation step. 
                     Smaller gamma values create a stronger watermark by enabling the watermarked model to achieve 
                     a greater differentiation from human/unwatermarked text because it is preferentially sampling 
                     from a smaller green set making those tokens less likely to occur by chance.
            - delta : The amount of positive bias to add to the logits of every token in the greenlist 
                        at each generation step before sampling/choosing the next token. Higher delta values 
                        mean that the greenlist tokens are more heavily preferred by the watermarked model
                        and as the bias becomes very large the watermark transitions from "soft" to "hard". 
                        For a hard watermark, nearly all tokens are green, but this can have a detrimental effect on
                        generation quality, especially when there is not a lot of flexibility in the distribution.

            #### Detector Parameters:
            
            - z-score threshold : the z-score cuttoff for the hypothesis test. Higher thresholds (such as 4.0) make
                                _false positives_ (predicting that human/unwatermarked text is watermarked) very unlikely
                                as a genuine human text with a significant number of tokens will almost never achieve 
                                that high of a z-score. Lower thresholds will capture more _true positives_ as some watermarked
                                texts will contain less green tokens and achive a lower z-score, but still pass the lower bar and 
                                be flagged as "watermarked". However, a lowere threshold will increase the chance that human text 
                                that contains a slightly higher than average number of green tokens is erroneously flagged. 
                                4.0-5.0 offers extremely low false positive rates while still accurately catching most watermarked text.
            - Ignore Bigram Repeats : This alternate detection algorithm only considers the unique bigrams in the text during detection, 
                                    computing the greenlists based on the first in each pair and checking whether the second falls within the list.
                                    This means that `T` is now the unique number of bigrams in the text, which becomes less than the total
                                    number of tokens generated if the text contains a lot of repetition. See the paper for a more detailed discussion.
            - Normalizations : we implement a few basic normaliations to defend against various adversarial perturbations of the
                                text analyzed during detection. Currently we support converting all chracters to unicode, 
                                replacing homoglyphs with a canonical form, and standardizing the capitalization. 
                                See the paper for a detailed discussion of input normalization. 
            """
            )
        
        gr.HTML("""
                <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. 
                    Follow the github link at the top and host the demo on your own GPU hardware to test out larger models.
                <br/>
                <a href="https://huggingface.co/spaces/tomg-group-umd/lm-watermarking?duplicate=true">
                <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                <p/>
                """)
        
        # Register main generation tab click, outputing generations as well as a the encoded+redecoded+potentially truncated prompt and flag
        generate_btn.click(fn=generate_ui_partial, inputs=[prompt,session_args], outputs=[redecoded_input, truncation_warning, output_without_watermark, output_with_watermark,session_args])
        # Show truncated version of prompt if truncation occurred
        redecoded_input.change(fn=truncate_prompt, inputs=[redecoded_input,truncation_warning,prompt,session_args], outputs=[prompt,session_args])
        # Call detection when the outputs (of the generate function) are updated
        output_without_watermark.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        output_with_watermark.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        # Register main detection tab click
        detect_btn.click(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result, session_args])

        # State management logic
        # update callbacks that change the state dict
        def update_sampling_temp(session_state, value): session_state.sampling_temp = float(value); return session_state
        def update_generation_seed(session_state, value): session_state.generation_seed = int(value); return session_state
        def update_gamma(session_state, value): session_state.gamma = float(value); return session_state
        def update_delta(session_state, value): session_state.delta = float(value); return session_state
        def update_detection_z_threshold(session_state, value): session_state.detection_z_threshold = float(value); return session_state
        def update_decoding(session_state, value):
            if value == "multinomial":
                session_state.use_sampling = True
            elif value == "greedy":
                session_state.use_sampling = False
            return session_state
        def toggle_sampling_vis(value):
            if value == "multinomial":
                return gr.update(visible=True)
            elif value == "greedy":
                return gr.update(visible=False)
        def toggle_sampling_vis_inv(value):
            if value == "multinomial":
                return gr.update(visible=False)
            elif value == "greedy":
                return gr.update(visible=True)
        def update_n_beams(session_state, value): session_state.n_beams = value; return session_state
        def update_max_new_tokens(session_state, value): session_state.max_new_tokens = int(value); return session_state
        def update_ignore_repeated_bigrams(session_state, value): session_state.ignore_repeated_bigrams = value; return session_state
        def update_normalizers(session_state, value): session_state.normalizers = value; return session_state
        def update_seed_separately(session_state, value): session_state.seed_separately = value; return session_state
        def update_select_green_tokens(session_state, value): session_state.select_green_tokens = value; return session_state
        # registering callbacks for toggling the visibilty of certain parameters
        decoding.change(toggle_sampling_vis,inputs=[decoding], outputs=[sampling_temp])
        decoding.change(toggle_sampling_vis,inputs=[decoding], outputs=[generation_seed])
        decoding.change(toggle_sampling_vis_inv,inputs=[decoding], outputs=[n_beams])
        # registering all state update callbacks
        decoding.change(update_decoding,inputs=[session_args, decoding], outputs=[session_args])
        sampling_temp.change(update_sampling_temp,inputs=[session_args, sampling_temp], outputs=[session_args])
        generation_seed.change(update_generation_seed,inputs=[session_args, generation_seed], outputs=[session_args])
        n_beams.change(update_n_beams,inputs=[session_args, n_beams], outputs=[session_args])
        max_new_tokens.change(update_max_new_tokens,inputs=[session_args, max_new_tokens], outputs=[session_args])
        gamma.change(update_gamma,inputs=[session_args, gamma], outputs=[session_args])
        delta.change(update_delta,inputs=[session_args, delta], outputs=[session_args])
        detection_z_threshold.change(update_detection_z_threshold,inputs=[session_args, detection_z_threshold], outputs=[session_args])
        ignore_repeated_bigrams.change(update_ignore_repeated_bigrams,inputs=[session_args, ignore_repeated_bigrams], outputs=[session_args])
        normalizers.change(update_normalizers,inputs=[session_args, normalizers], outputs=[session_args])
        seed_separately.change(update_seed_separately,inputs=[session_args, seed_separately], outputs=[session_args])
        select_green_tokens.change(update_select_green_tokens,inputs=[session_args, select_green_tokens], outputs=[session_args])
        # register additional callback on button clicks that updates the shown parameters window
        generate_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detect_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        # When the parameters change, display the update and fire detection, since some detection params dont change the model output.
        gamma.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        gamma.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        gamma.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        gamma.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        detection_z_threshold.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detection_z_threshold.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        detection_z_threshold.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        detection_z_threshold.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        ignore_repeated_bigrams.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        normalizers.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        normalizers.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        normalizers.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        normalizers.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        select_green_tokens.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        select_green_tokens.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        select_green_tokens.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        select_green_tokens.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])


    demo.queue()

    if args.demo_public:
        demo.launch(share=True) # exposes app to the internet via randomly generated link
    else:
        demo.launch()

def main(args): 
    """Run the generation and detection operations
    to collect experimental data using a local CSV file, and save 
    watermarked and non-watermarked results to separate CSV files.
    Optionally launches gradio demo."""
    
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args) 
    else:
        model, tokenizer, device = None, None, None

    # --- 데이터 수집 모드 시작: 로컬 CSV 파일 사용 ---
    
    # 1. 로컬 CSV 파일 설정
    INPUT_CSV_FILENAME = "human_prompts.csv"
    PROMPT_COLUMN = 'data'    # 프롬프트가 포함된 CSV 컬럼 이름 (필요에 따라 변경)
    MAX_PROMPTS = 100         # 테스트를 위해 최대 100개만 사용하도록 제한
    
    # 출력 파일명 설정
    output_filename_no_wm = "experiment_data_no_wm.csv"
    output_filename_wm = "experiment_data_wm.csv"
    
    prompts = []
    
    try:
        print(f"Loading prompts from local CSV file: {INPUT_CSV_FILENAME}...")
        with open(INPUT_CSV_FILENAME, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                # PROMPT_COLUMN이 있고, 그 값이 비어있지 않은 경우에만 추가
                if PROMPT_COLUMN in row and row[PROMPT_COLUMN].strip():
                    prompts.append(row[PROMPT_COLUMN].strip())
                if len(prompts) >= MAX_PROMPTS:
                    print(f"Reached max prompts limit of {MAX_PROMPTS}.")
                    break
        
        if not prompts:
             print(f"No valid prompts found in '{INPUT_CSV_FILENAME}' in column '{PROMPT_COLUMN}'.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_FILENAME}' was not found.")
        prompts = [] # 파일이 없으면 프롬프트 리스트를 비움
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        prompts = []

    # 프롬프트가 성공적으로 로드된 경우에만 생성 및 탐지 작업 수행
    if prompts and not args.skip_model_load:
        print(f"Loaded {len(prompts)} prompts from '{INPUT_CSV_FILENAME}'. Starting data generation...")
        
        # CSV 파일 헤더 정의
        fieldnames = ["id", "original_prompt", "generated_text", "p_value", "z_score"]

        # CSV 파일 두 개를 동시에 엽니다.
        with open(output_filename_no_wm, 'w', newline='', encoding='utf-8') as csvfile_no_wm, \
             open(output_filename_wm, 'w', newline='', encoding='utf-8') as csvfile_wm:
            
            writer_no_wm = csv.DictWriter(csvfile_no_wm, fieldnames=fieldnames)
            writer_wm = csv.DictWriter(csvfile_wm, fieldnames=fieldnames)
            
            writer_no_wm.writeheader()
            writer_wm.writeheader()

            for i, input_text in enumerate(prompts):
                print(f"[{i+1}/{len(prompts)}] Processing prompt...")

                # 워터마크 없는 텍스트 생성 및 탐지
                decoded_output_without_watermark= generate(input_text, 
                                                                         args, 
                                                                         model=model, 
                                                                         device=device, 
                                                                         tokenizer=tokenizer,
                                                                         is_watermarked=False) 

                score_no_wm = detect_raw(decoded_output_without_watermark, args, device=device, tokenizer=tokenizer)

                no_wm_p_value = score_no_wm.get('p_value', "N/A")
                no_wm_z_score = score_no_wm.get('z_score', "N/A")

                
                # No-WM 결과 파일에 작성
                row_no_wm = {
                    "id": i,
                    "original_prompt": input_text,
                    "generated_text": decoded_output_without_watermark.strip(),
                    "p_value": no_wm_p_value,
                    "z_score": no_wm_z_score,
                }
                writer_no_wm.writerow(row_no_wm)

                # 워터마크 있는 텍스트 생성 및 탐지
                decoded_output_with_watermark = generate(
                    input_text,
                    args,
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    is_watermarked=True
                )

                score_wm = detect_raw(decoded_output_with_watermark, args, device=device, tokenizer=tokenizer)

                wm_p_value = score_wm.get('p_value', "N/A")
                wm_z_score = score_wm.get('z_score', "N/A")


                # WM 결과 파일에 작성
                row_wm = {
                    "id": i,
                    "original_prompt": input_text,
                    "generated_text": decoded_output_with_watermark.strip(),
                    "p_value": wm_p_value,
                    "z_score": wm_z_score,
                }
                writer_wm.writerow(row_wm)

                print(f"  -> WM Z-score: {wm_z_score}, No WM Z-score: {no_wm_z_score}")
                print("-" * 50)
            
        print(f"\n--- Data generation complete. Saved {len(prompts)} rows to {output_filename_no_wm} and {output_filename_wm} ---")
        
    elif args.skip_model_load:
        print("Skipping generation: Model loading was skipped.")
        
    # --- Gradio 데모 실행 부분 ---
    if args.run_gradio:
        print("\nLaunching Gradio demo...")
        run_gradio(args, model=model, tokenizer=tokenizer, device=device)

    return

if __name__ == "__main__":

    args = parse_args()
    print(args)

    main(args)