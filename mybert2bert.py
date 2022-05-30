from lightningbase import LightningBase
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
    BertModel,
)
import torch

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MyB2BGenerator(LightningBase):
    def __init__(
        self,
        model_save_path: str,
        max_len: int = 512,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        save_step_interval: int = 1000,
    ) -> None:
        super(MyB2BGenerator, self).__init__(
            model_save_path=model_save_path,
            max_len=max_len,
            lr=lr,
            weight_decay=weight_decay,
            save_step_interval=save_step_interval,
        )

        encoder_config = BertConfig.from_pretrained("monologg/kobert")
        decoder_config = BertConfig.from_pretrained("monologg/kobert")
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )

        self.model = EncoderDecoderModel(config)
        state_dict = BertModel.from_pretrained("monologg/kobert").state_dict()
        self.model.encoder.load_state_dict(state_dict)
        self.model.decoder.bert.load_state_dict(state_dict, strict=False)
        # cross attention이랑 lm head는 처음부터 학습
        self.pad_token_id = 1

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        
        mask_tensor = torch.ones(input_ids.size()).long().to(self.device)
        attention_mask = torch.where(input_ids == 1, input_ids, mask_tensor * -1).long()
        attention_mask = torch.where(attention_mask == -1, attention_mask, mask_tensor * 0).long()
        attention_mask = torch.where(attention_mask != -1, attention_mask, mask_tensor).long().to(self.device)

        labels = batch['labels'].to(self.device)

        decoder_input_ids = shift_tokens_right(labels, 1,2).to(self.device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        lm_logits = outputs[0]
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=1
        )

        lm_loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        self.save_model()
        return {"loss": lm_loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        
        mask_tensor = torch.ones(input_ids.size()).long().to(self.device)
        attention_mask = torch.where(input_ids == 1, input_ids, mask_tensor * -1).long()
        attention_mask = torch.where(attention_mask == -1, attention_mask, mask_tensor * 0).long()
        attention_mask = torch.where(attention_mask != -1, attention_mask, mask_tensor).long().to(self.device)

        labels = batch['labels'].to(self.device)

        decoder_input_ids = shift_tokens_right(labels, 1,2).to(self.device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        lm_logits = outputs[0]
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=1
        )

        lm_loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        self.save_model()
        self.log("val_loss", lm_loss)
        return {"val_loss": lm_loss}
