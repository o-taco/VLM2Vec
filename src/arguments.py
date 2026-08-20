from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    model_backbone: str = field(
        default=None,
        metadata={"help": "backbone name"}
    )
    processor_name: str = field(
        default=None, metadata={"help": "processor_name, huggingface model name or path"}
    )
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path"}
    )
    pooling: str = field(
        default='last',
        metadata={"help": "pooling method for encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "temperature for softmax"}
    )
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    lora_layers_to_transform: str = field(
        default=None,
        metadata={"help": "comma-separated decoder layer indices to attach LoRA adapters to (e.g. '21,22,23,24,25,26,27'). "
                           "Layers not listed keep their pretrained weights frozen but still run forward (unlike depth "
                           "pruning, which removes them). Default (unset) attaches LoRA to every layer, the prior behavior."}
    )
    add_ffn_head: bool = field(
        default=False,
        metadata={"help": "attach a small residual 2-layer FFN (Linear-GELU-Linear) after pooling, before "
                           "normalization, trained jointly with the contrastive loss. Combined with "
                           "--freeze_backbone and --lora False, isolates how much of the training gain comes from "
                           "reshaping the readout embedding space rather than adapting internal representations. "
                           "NOTE: despite being called a 'linear probe' in some places, this head is nonlinear "
                           "(has a GELU) -- use --add_linear_head for a genuine linear probe."}
    )
    add_linear_head: bool = field(
        default=False,
        metadata={"help": "attach a single residual nn.Linear (no hidden layer, no activation) after pooling, "
                           "before normalization -- a genuine linear probe, unlike --add_ffn_head's 2-layer "
                           "nonlinear MLP. Mutually exclusive in practice with --add_ffn_head; reuses "
                           "--ffn_residual for the zero-init/identity-start behavior."}
    )
    ffn_hidden_dim: int = field(
        default=None,
        metadata={"help": "hidden width of the FFN head's middle layer; defaults to the backbone hidden size"}
    )
    ffn_residual: bool = field(
        default=True,
        metadata={"help": "if True, the head output is added to the pooled reps (zero-initialized, so training "
                           "starts at the frozen zero-shot embedding); if False, the head output replaces them"}
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "freeze all encoder weights (must be used with --lora False); pairs with --add_ffn_head "
                           "to train only the readout head on top of a fully frozen backbone"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "number of crops used in image encoder"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    split_name: List[str] = field(
        default='original', metadata={"help": "'original', 'diverse_instruction'"}
    )
    subset_name: List[str] = field(
        default=None, metadata={"help": "Useful for datasets with subsets"}
    )
    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )
    num_sample_per_subset: int = field(
        default=100, metadata={"help": "number of training samples per subset"}
    )
    image_dir: str = field(
        default=None, metadata={"help": "Image directory path"}
    )
    encode_output_path: str = field(
        default=None, metadata={"help": "encode output path"}
    )
    max_len: int = field(
        default=None, metadata={"help": "The maximum total input sequence length after tokenization. "
                                        "Use with caution, since it may truncate text prompts due to large image lengths."},
    )
    embedding_type: str = field(
        default="", metadata={"help": "embedding type"}
    )
    image_resolution: str = field(
        default='high', metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    image_encoder_freeze: bool = field(
        default=False, metadata={"help": "huggingface model name"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    project_name: str = field(
        default=None, metadata={"help": "project name"}
    )

    logging_steps: int = field(
        default=1, metadata={"help": "logging steps"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(
        default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(
        default=2, metadata={"help": "target side subset size"})


@dataclass
class MTEBArguments:
    task_types: List[str] = field(
        default=None, metadata={"help": ""}
    )
    tasks: List[str] = field(
        default=None, metadata={"help": ""}
    )
