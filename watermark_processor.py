from __future__ import annotations
import collections
from math import sqrt

import scipy.stats
import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor
from nltk.util import ngrams
from normalizers import normalization_strategy_lookup


import numpy as np
from collections import defaultdict



class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        hash_key: int = 15485863,
        select_green_tokens: bool = True,
        cluster_data_path: str = "cluster_data.npz",
        tokenizer: Tokenizer = None,  

        cluster_gamma: float = 0.15, # 클러스터 분할 방식을 사용할 "확률" (0.0 ~ 1.0)
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma # 
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        
        self.cluster_gamma = cluster_gamma 

        assert tokenizer is not None, "Clustering mode requires a tokenizer instance."
        self.tokenizer = tokenizer

        print(f"Loading cluster data from: {cluster_data_path}")
        try:
            # 1. allow_pickle=True로 로드
            cluster_data = np.load(cluster_data_path, allow_pickle=True)
            
            token_strings = cluster_data['tokens'] 
            labels = cluster_data['labels']

            # 2. 중복을 피하기 위해 set을 사용하는 defaultdict 생성
            cluster_map_sets = defaultdict(set)

            print("Mapping token strings to tokenizer vocab IDs...")
            skipped_count = 0
            
            for token_str, label in zip(token_strings, labels):
                
                # 3. 토크나이저를 사용해 문자열을 정수 ID 리스트로 변환
                token_id_list = self.tokenizer.encode(str(token_str), add_special_tokens=False)
                
                # 4. ID 리스트가 비어있지 않다면,
                if token_id_list:
                    # 'update'를 사용해 리스트의 모든 ID를 set에 추가
                    cluster_map_sets[label].update(token_id_list)
                else:
                    skipped_count += 1

            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} entries that tokenized to an empty list.")

            # 5.set 맵을 최종 list 맵으로 변환
            self.cluster_map = {
                label: list(token_set) for label, token_set in cluster_map_sets.items()
            }

            self.unique_labels = sorted(self.cluster_map.keys())
            self.num_clusters = len(self.cluster_map) #
            
            # 이 숫자가 0이 아니어야 합니다.
            print(f"Successfully loaded {self.num_clusters} clusters.")
            if self.num_clusters == 0:
                print("CRITICAL WARNING: No clusters were loaded. Check .npz file content and tokenizer matching.")

        except FileNotFoundError:
            print(f"Error: Cluster data file '{cluster_data_path}' not found.")
            raise
        except KeyError as e:
            print(f"Error: .npz file missing required array: {e}")
            raise

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:

        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    # --- Step 2: 확률에 따라 두 가지 분할 방식을 선택 ---
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """
        (수정된 메소드)
        self.cluster_gamma 확률에 따라 '클러스터-인식' 방식과
        '토큰-인식' 방식 중 하나를 선택하여 그린 리스트를 생성합니다.
        두 방식 모두 self.gamma를 분할 비율로 사용합니다.
        """
        
        # 1. 컨텍스트 기반으로 시드 설정 
        self._seed_rng(input_ids)

        # 2. 분할 방식을 결정하기 위한 난수 생성
        #    이 난수 생성은 시드에 따라 결정적이므로, Processor와 Detector가 동일하게 작동합니다.
        
        decision_roll = torch.rand(1, generator=self.rng, device=input_ids.device).item() # 0.0 ~ 1.0 사이의 값


        if decision_roll < self.cluster_gamma:
            # --- 방식 1: (cluster_gamma 확률) "클러스터-인식" 분할 ---
            # (이전에 만든 클러스터 분할 로직)
            
            # 2-1. "그린 클러스터"의 개수를 self.gamma 기준으로 계산
            green_cluster_count = int(self.num_clusters * self.gamma) 

            # 2-2. 클러스터를 셔플 (device=input_ids.device가 이미 올바르게 적용되어 있음)
            cluster_permutation = torch.randperm(self.num_clusters, device=input_ids.device, generator=self.rng)
            
            # 2-3. "그린 클러스터" 인덱스 선택
            if self.select_green_tokens:
                green_cluster_indices = cluster_permutation[:green_cluster_count]
            else:
                green_cluster_indices = cluster_permutation[(self.num_clusters - green_cluster_count) :]
            
            # 2-4. 레이블 조회
            green_labels = [self.unique_labels[i] for i in green_cluster_indices.cpu().tolist()]

            # 2-5. 토큰 ID로 확장
            greenlist_token_ids = []
            for label in green_labels:
                greenlist_token_ids.extend(self.cluster_map[label])
        
        else:
            # --- 방식 2: (1 - cluster_gamma 확률) "토큰-인식" 분할 ---
            # (가장 처음의 원본 코드 로직)
            
            # 2-1. "그린 토큰"의 개수를 self.gamma 기준으로 계산
            greenlist_size = int(self.vocab_size * self.gamma)
            
            # 2-2. 전체 어휘를 셔플 (device=input_ids.device가 이미 올바르게 적용되어 있음)
            vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
            
            # 2-3. "그린 토큰" 인덱스 선택
            if self.select_green_tokens:
                greenlist_ids = vocab_permutation[:greenlist_size]
            else:
                greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]
            
            # 2-4. Python 리스트로 변환
            greenlist_token_ids = greenlist_ids.cpu().tolist()

        return greenlist_token_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        # *args와 **kwargs를 통해 cluster_gamma 등의 모든 부모 인자를
        # WatermarkBase.__init__으로 자동 전달합니다.
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            # greenlist_token_ids[b_idx]는 이제 확률적으로
            # (클러스터 확장 리스트) 또는 (개별 토큰 리스트)가 됩니다.
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            # self._get_greenlist_ids가 이제 확률적 분할 로직을 수행합니다.
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        # WatermarkBase가 클러스터링을 위해 tokenizer를 필요로 하므로,
        # super() 호출 시 명시적으로 전달해야 합니다.
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        # self._get_greenlist_ids()는 Processor와
        # 동일한 "확률적 분할"을 수행합니다.
        
        if self.ignore_repeated_bigrams:
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict