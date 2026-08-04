import json
import os
from typing import Dict, Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments
from src.model_utils import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, QWEN3_VL, backbone2model
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.vlm_backbone.qwen3_vl import patch_vision_patch_embed_for_volta

FFN_HEAD_WEIGHTS_NAME = 'ffn_head.pt'
FFN_HEAD_CONFIG_NAME = 'ffn_head_config.json'


def _get_hidden_size(config) -> int:
    return getattr(config, 'hidden_size', None) or config.text_config.hidden_size


class FFNHead(nn.Module):
    """Small readout head applied to the pooled embedding, after pooling and
    before normalization. In residual mode the last linear layer is
    zero-initialized so the head starts as an exact identity function --
    training begins at the frozen zero-shot embedding and the head learns an
    additive correction, isolating "adapting the embedding space" from
    "adapting internal representations" (LoRA)."""

    def __init__(self, dim: int, hidden_dim: int, residual: bool = True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return x + out if self.residual else out


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.ffn_head = None
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def add_ffn_head(self, hidden_dim: int = None, residual: bool = True) -> FFNHead:
        dim = _get_hidden_size(self.config)
        param = next(self.encoder.parameters())
        self.ffn_head = FFNHead(dim, hidden_dim or dim, residual=residual).to(device=param.device, dtype=param.dtype)
        return self.ffn_head

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, input):
        # grad_cache's manual chunked forward bypasses Trainer's usual input-to-device
        # step, so collator output (CPU tensors) never gets moved to the model's device.
        device = next(self.encoder.parameters()).device
        input = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input.items()}
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        reps = self._pool(hidden_states, input['attention_mask'])
        if self.ffn_head is not None:
            reps = self.ffn_head(reps)
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments=None, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == QWEN3_VL:
            # Volta (sm_70, e.g. V100) has no FlashAttention-2 support; sdpa runs everywhere.
            config._attn_implementation = "sdpa"
            config.vision_config._attn_implementation = "sdpa"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            patch_vision_patch_embed_for_volta(base_model)
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            layers_to_transform = None
            if model_args.lora_layers_to_transform:
                layers_to_transform = [int(i) for i in model_args.lora_layers_to_transform.split(',')]
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                layers_to_transform=layers_to_transform,
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                # DoRA causes a no-grad/dead-gradient bug on this training path (found empirically
                # in the prior Colab run on the V2 codebase); plain LoRA trains correctly.
                use_dora=False,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            if model_args.freeze_backbone:
                for p in base_model.parameters():
                    p.requires_grad_(False)
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        if model_args.add_ffn_head:
            model.add_ffn_head(hidden_dim=model_args.ffn_hidden_dim, residual=model_args.ffn_residual)

        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=False, **kwargs):
        # Loading the base model
        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')

        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN3_VL}:
            # Volta (sm_70, e.g. V100) has no FlashAttention-2 support; sdpa runs everywhere.
            config._attn_implementation = "sdpa"
            config.vision_config._attn_implementation = "sdpa"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                config=config
            )
            if model_args.model_backbone == QWEN3_VL:
                patch_vision_patch_embed_for_volta(base_model)
        elif model_args.model_backbone == PHI3V:
            # Loading the base model
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                checkpoint_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # Building the model on top of the base
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config,
                                                   is_trainable=is_trainable)
            lora_model.load_adapter(checkpoint_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        ffn_head_config_path = os.path.join(checkpoint_path, FFN_HEAD_CONFIG_NAME)
        if os.path.exists(ffn_head_config_path):
            with open(ffn_head_config_path) as f:
                head_cfg = json.load(f)
            model.add_ffn_head(hidden_dim=head_cfg['hidden_dim'], residual=head_cfg['residual'])
            state_dict = torch.load(os.path.join(checkpoint_path, FFN_HEAD_WEIGHTS_NAME), map_location='cpu', weights_only=True)
            model.ffn_head.load_state_dict(state_dict)
            if not is_trainable:
                model.ffn_head.eval()

        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        if self.ffn_head is not None:
            torch.save(self.ffn_head.state_dict(), os.path.join(output_dir, FFN_HEAD_WEIGHTS_NAME))
            with open(os.path.join(output_dir, FFN_HEAD_CONFIG_NAME), 'w') as f:
                json.dump({'hidden_dim': self.ffn_head.hidden_dim, 'residual': self.ffn_head.residual}, f)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
