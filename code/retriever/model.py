from common.utils import IMAGE_SIZE, local_device

MAX_LENGTH = 8192
INSTRUCTION = "Represent the user's input."


class Qwen3VLEmbedding:
    def __init__(self, model_path, device):
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local embedding model directory not found: {model_path}")

        import torch
        from transformers import Qwen3VLModel, Qwen3VLProcessor

        self.torch = torch
        self.device, dtype = local_device(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path, local_files_only=True, padding_side="right",
        )
        model, info = Qwen3VLModel.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True,
            # Qwen saves the backbone under "model." in its embedding checkpoints.
            key_mapping={r"^model\.": ""},
            dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
        )
        if info["missing_keys"]:
            raise ValueError(f"Incomplete embedding checkpoint: {info['missing_keys'][:5]}")
        self.model = model.to(self.device).eval()

    encoding = [MAX_LENGTH, INSTRUCTION, IMAGE_SIZE]

    def encode_batch(self, items, role):
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {"role": "user", "content": [item]},
            ]
            for item in items
        ]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        images = [item["image"] for item in items if item["type"] == "image"]
        inputs = self.processor(
            text=text, images=images or None, images_kwargs={"size": IMAGE_SIZE},
            padding=True, padding_side="right", truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        torch = self.torch
        with torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            last = inputs.attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(len(items), device=self.device), last].float()
            vectors = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
        return vectors


class JinaEmbeddingsV4:
    encoding = ["retrieval", "single-vector", 8192]

    def __init__(self, model_path, device):
        import torch
        from transformers import AutoModel

        self.torch = torch
        device, dtype = local_device(device)
        self.model = AutoModel.from_pretrained(
            str(model_path.resolve()), local_files_only=True, trust_remote_code=True,
            torch_dtype=dtype, attn_implementation="sdpa", use_safetensors=True,
        ).to(device).eval()

    def encode_batch(self, items, role):
        options = {"task": "retrieval", "batch_size": len(items), "return_multivector": False}
        if role == "image":
            vectors = self.model.encode_image([item["image"] for item in items], **options)
        else:
            vectors = self.model.encode_text(
                [item["text"] for item in items], prompt_name=role, max_length=8192, **options,
            )
        return self.torch.stack(vectors).detach().float().cpu().numpy()


class WeMMEmbedding:
    encoding = ["embedding", "right-padding", 16]

    def __init__(self, model_path, device):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.device, dtype = local_device(device)
        options = {"local_files_only": True, "trust_remote_code": True}
        self.processor = AutoProcessor.from_pretrained(model_path, **options)
        self.model = AutoModel.from_pretrained(
            model_path, dtype=dtype, use_safetensors=True, **options,
        ).to(self.device).eval()

    def encode_batch(self, items, role):
        messages = [[{"role": "user", "content": [item]}] for item in items]
        texts = [
            self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in messages
        ]
        images, _ = self.process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=texts, images=images, padding=True, padding_side="right", return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            return self.model.embedding(**inputs, use_cache=False).float().cpu().numpy()


DEFAULT_EMBEDDER = "Qwen3-VL-Embedding-8B"
EMBEDDERS = {
    DEFAULT_EMBEDDER: Qwen3VLEmbedding,
    "jina-embeddings-v4": JinaEmbeddingsV4,
    "WeMM-Embedding-9B": WeMMEmbedding,
}
