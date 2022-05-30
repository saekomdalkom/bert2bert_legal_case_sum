import json
import sys
import pandas as pd

import torch
from tqdm import tqdm

from transformers import EncoderDecoderConfig, BertConfig, EncoderDecoderModel
from kobert_transformers import get_tokenizer


@torch.no_grad()
def inference():
    step = sys.argv[1]
    encoder_config = BertConfig.from_pretrained("monologg/kobert")
    decoder_config = BertConfig.from_pretrained("monologg/kobert")
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )

    tokenizer = KoBertTokenizer()
    model = EncoderDecoderModel(config=config)
    ckpt = "model.pt"
    device = "cuda"

    model.load_state_dict(
        torch.load(
            f"saved/{ckpt}.{step}", map_location="cuda"
        ),
        strict=True,
    )

    model = model.half().eval().to(device)
    test_data = open("data/test.jsonl", "r").read().splitlines()
    # test_data = open("dataset/id_issue_finding_small_test.jsonl", "r").read().splitlines()

    # submission = open(f"submission_{step}.csv", "w", encoding='utf-8-sig')
    submission = []

    test_set = []
    for data in test_data:
        data = json.loads(data)
        article_original = data["article_original"]
        article_original = " ".join(article_original)
        news_id = data["id"]
        test_set.append((news_id, article_original))

    for i, (news_id, text) in tqdm(enumerate(test_set)):
        tokens = tokenizer.encode_batch([text], max_length=512)
        generated = model.generate(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
            use_cache=True,
            bos_token_id=tokenizer.token2idx["[CLS]"],
            eos_token_id=tokenizer.token2idx["[SEP]"],
            pad_token_id=tokenizer.token2idx["[PAD]"],
            num_beams=12,
            do_sample=False,
            temperature=1.0,
            no_repeat_ngram_size=4,
            bad_words_ids=[[tokenizer.token2idx["[UNK]"]]],
            length_penalty=1.5,
            max_length=512,
        )

        output = tokenizer.decode_batch(generated.tolist())[0]
        # submission.write(f"{news_id},{output}" + "\n")
        case_dict = {}
        case_dict['id'] = news_id
        case_dict['output'] = output
        submission.append(case_dict)
        print(news_id, output)
    case_df = pd.DataFrame(submission)
    case_df.to_csv(f"submission_test_{step}.csv", encoding='utf-8-sig', index=False)



class KoBertTokenizer(object):
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.token2idx = self.tokenizer.token2idx
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode_batch(self, x, max_length):
        max_len = 0
        result_tokenization = []

        for i in x:
            tokens = self.tokenizer.encode(i, max_length=max_length, truncation=True)
            result_tokenization.append(tokens)   # tokens: list of integers. batch 중에서 input 한 문장.

            if len(tokens) > max_len:
                max_len = len(tokens)

        padded_tokens = []
        for tokens in result_tokenization:
            padding = (torch.ones(max_len) * self.token2idx["[PAD]"]).long()
            padding[: len(tokens)] = torch.tensor(tokens).long()
            padded_tokens.append(padding.unsqueeze(0))

        padded_tokens = torch.cat(padded_tokens, dim=0).long()
        mask_tensor = torch.ones(padded_tokens.size()).long()

        attention_mask = torch.where(
            padded_tokens == self.token2idx["[PAD]"], padded_tokens, mask_tensor * -1
        ).long()
        attention_mask = torch.where(
            attention_mask == -1, attention_mask, mask_tensor * 0
        ).long()
        attention_mask = torch.where(
            attention_mask != -1, attention_mask, mask_tensor
        ).long()

        return {
            "input_ids": padded_tokens.long(),
            "attention_mask": attention_mask.long(),
        }

    def decode(self, tokens):   # tokens: list of integers. batch 중에서 input 한 문장.
        # remove special tokens
        # unk, pad, cls, sep, mask
        tokens = [token for token in tokens
                  if token not in [0, 1, 2, 3, 4]]

        decoded = [self.idx2token[token] for token in tokens]
        if "▁" in decoded[0] and "▁" in decoded[1]:
            # fix decoding bugs
            tokens = tokens[1:]

        return self.tokenizer.decode(tokens)

    def decode_batch(self, list_of_tokens):
        return [self.decode(tokens) for tokens in list_of_tokens]





if __name__ == '__main__':
    inference()
