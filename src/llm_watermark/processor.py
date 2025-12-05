from __future__ import annotations
import collections
from math import sqrt
import os

import scipy.stats
import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor
from nltk.util import ngrams
from .normalizers import normalization_strategy_lookup

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
        cluster_data_path: str = None,   # ⭐ 변경됨 (None = 자동 경로)
        tokenizer: Tokenizer = None,
        cluster_gamma: float = 0.15,
    ):

        # watermark parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.cluster_gamma = cluster_gamma

        assert tokenizer is not None, "Clustering mode requires a tokenizer instance."
        self.tokenizer = tokenizer

        # ⭐ cluster_data_path 자동 설정 ⭐
        if cluster_data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))   # llm_watermark/
            cluster_data_path = os.path.join(base_dir, "data", "cluster_data.npz")

        print(f"Loading cluster data from: {cluster_data_path}")
        try:
            cluster_data = np.load(cluster_data_path, allow_pickle=True)
            token_strings = cluster_data['tokens']
            labels = cluster_data['labels']

            cluster_map_sets = defaultdict(set)
            print("Mapping token strings to tokenizer vocab IDs...")
            skipped_count = 0

            for token_str, label in zip(token_strings, labels):
                token_id_list = self.tokenizer.encode(str(token_str), add_special_tokens=False)
                if token_id_list:
                    cluster_map_sets[label].update(token_id_list)
                else:
                    skipped_count += 1

            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} entries that tokenized to an empty list.")

            self.cluster_map = {
                label: list(token_set)
                for label, token_set in cluster_map_sets.items()
            }

            self.unique_labels = sorted(self.cluster_map.keys())
            self.num_clusters = len(self.cluster_map)

            print(f"Successfully loaded {self.num_clusters} clusters.")
            if self.num_clusters == 0:
                print("CRITICAL WARNING: No clusters loaded. Check .npz content or tokenizer mismatch.")

        except FileNotFoundError:
            print(f"Error: Cluster data file '{cluster_data_path}' not found.")
            raise
        except KeyError as e:
            print(f"Error: .npz file missing required array: {e}")
            raise

    # ============================================================
    # RANDOM SEEDING
    # ============================================================
    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, "simple_1 requires ≥1 prefix tokens"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    # ============================================================
    # GREENLIST SELECTION
    # ============================================================
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        self._seed_rng(input_ids)
        decision_roll = torch.rand(1, generator=self.rng, device=input_ids.device).item()

        # ---- 1) CLUSTER-BASED SPLIT ----
        if decision_roll < self.cluster_gamma:
            green_cluster_count = int(self.num_clusters * self.gamma)
            cluster_perm = torch.randperm(self.num_clusters, device=input_ids.device, generator=self.rng)

            if self.select_green_tokens:
                green_indices = cluster_perm[:green_cluster_count]
            else:
                green_indices = cluster_perm[(self.num_clusters - green_cluster_count):]

            green_labels = [self.unique_labels[i] for i in green_indices.cpu().tolist()]

            greenlist_token_ids = []
            for label in green_labels:
                greenlist_token_ids.extend(self.cluster_map[label])

        # ---- 2) TOKEN-BASED SPLIT ----
        else:
            green_size = int(self.vocab_size * self.gamma)
            vocab_perm = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)

            if self.select_green_tokens:
                green_ids = vocab_perm[:green_size]
            else:
                green_ids = vocab_perm[(self.vocab_size - green_size):]

            greenlist_token_ids = green_ids.cpu().tolist()

        return greenlist_token_ids


# ============================================================
# PROCESSOR
# ============================================================
class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids):
        green_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        return green_mask.bool()

    def _bias_greenlist_logits(self, scores, mask, bias):
        scores[mask] = scores[mask] + bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        batch_green_ids = []
        for b_idx in range(input_ids.shape[0]):
            batch_green_ids.append(self._get_greenlist_ids(input_ids[b_idx]))

        mask = self._calc_greenlist_mask(scores, batch_green_ids)
        scores = self._bias_greenlist_logits(scores, mask, self.delta)
        return scores


# ============================================================
# DETECTOR
# ============================================================
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
        super().__init__(*args, tokenizer=tokenizer, **kwargs)

        assert device, "Must pass device"
        assert tokenizer, "Need tokenizer instance for detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.min_prefix_len = 1  # only simple_1 supported
        self.normalizers = [normalization_strategy_lookup(n) for n in normalizers]

        self.ignore_repeated_bigrams = ignore_repeated_bigrams

    # Z-score
    def _compute_z_score(self, observed_count, T):
        expected = self.gamma
        num = observed_count - expected * T
        den = sqrt(T * expected * (1 - expected))
        return num / den

    def _compute_p_value(self, z):
        return scipy.stats.norm.sf(z)

    # MAIN WATERMARK SCORING
    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored=True,
        return_num_green_tokens=True,
        return_green_fraction=True,
        return_green_token_mask=False,
        return_z_score=True,
        return_p_value=True,
    ):

        if self.ignore_repeated_bigrams:
            bigram_table = {}
            bigrams = list(ngrams(input_ids.cpu().tolist(), 2))
            T = len(bigrams)

            for bg in bigrams:
                prefix = torch.tensor([bg[0]], device=self.device)
                green = self._get_greenlist_ids(prefix)
                bigram_table[bg] = (bg[1] in green)

            green_count = sum(bigram_table.values())

        else:
            T = len(input_ids) - self.min_prefix_len
            green_count = 0

            for idx in range(self.min_prefix_len, len(input_ids)):
                prefix = input_ids[:idx]
                green_list = self._get_greenlist_ids(prefix)
                if input_ids[idx] in green_list:
                    green_count += 1

        result = {}
        if return_num_tokens_scored:
            result["num_tokens_scored"] = T
        if return_num_green_tokens:
            result["num_green_tokens"] = green_count
        if return_green_fraction:
            result["green_fraction"] = green_count / T

        if return_z_score:
            z = self._compute_z_score(green_count, T)
            result["z_score"] = z
        if return_p_value:
            z = result.get("z_score", self._compute_z_score(green_count, T))
            result["p_value"] = self._compute_p_value(z)

        return result

    # ============================================================
    # DETECT
    # ============================================================
    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction=True,
        return_scores=True,
        z_threshold=None,
        **kwargs,
    ):
        assert (text is not None) ^ (tokenized_text is not None), "Pass either raw text or IDs"

        if return_prediction:
            kwargs["return_p_value"] = True

        # normalization
        for norm in self.normalizers:
            text = norm(text)

        if tokenized_text is None:
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        else:
            ids = tokenized_text.to(self.device)

        # remove BOS
        if ids[0] == self.tokenizer.bos_token_id:
            ids = ids[1:]

        score_dict = self._score_sequence(ids, **kwargs)

        out = {}
        if return_scores:
            out.update(score_dict)

        if return_prediction:
            thr = z_threshold if z_threshold is not None else self.z_threshold
            out["prediction"] = score_dict["z_score"] > thr
            if out["prediction"]:
                out["confidence"] = 1 - score_dict["p_value"]

        return out
