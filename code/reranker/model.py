from common.utils import IMAGE_SIZE, local_device

# https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
MAX_LENGTH = 10240


class Qwen3VLReranker:
    def __init__(self, model_path, device):
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local reranker model directory not found: {model_path}")

        import torch
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        self.torch = torch
        self.device, dtype = local_device(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path, local_files_only=True, padding_side="left",
        )
        model, info = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True,
            dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
        )
        if info["missing_keys"]:
            raise ValueError(f"Incomplete reranker checkpoint: {info['missing_keys'][:5]}")

        vocab = self.processor.tokenizer.get_vocab()
        weights = model.lm_head.weight.detach()
        # The difference of these rows gives logit(yes) - logit(no).
        self.score_weight = (weights[vocab["yes"]].float() - weights[vocab["no"]].float()).to(self.device)
        self.model = model.model.to(self.device).eval()

    def score_batch(self, query, images):
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": f"<Instruct>: {INSTRUCTION}<Query>:{query or 'NULL'}\n<Document>:"},
                    {"type": "image", "image": image},
                ]},
            ]
            for image in images
        ]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(
            text=text, images=images, images_kwargs={"size": IMAGE_SIZE},
            padding=True, padding_side="left", return_tensors="pt",
        )
        if inputs.input_ids.shape[1] > MAX_LENGTH:
            raise ValueError(f"Reranker input exceeds {MAX_LENGTH} tokens")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state[:, -1]
            return (hidden.float() @ self.score_weight).sigmoid().tolist()


class JinaRerankerM0:
    def __init__(self, model_path, device):
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local reranker model directory not found: {model_path}")

        from transformers import AutoModel, AutoProcessor

        device, dtype = local_device(device)
        options = {"local_files_only": True, "trust_remote_code": True}
        self.model = AutoModel.from_pretrained(
            str(model_path.resolve()), dtype=dtype, attn_implementation="sdpa",
            use_safetensors=True, **options,
        ).to(device).eval()
        # Initialize the processor explicitly so the official scorer uses local files.
        self.model._processor = AutoProcessor.from_pretrained(
            model_path, max_pixels=602112, min_pixels=3136, **options,
        )
        self.model._processor.tokenizer.padding_side = "left"

    def score_batch(self, query, images):
        scores = self.model.compute_score(
            [[query or "NULL", image] for image in images],
            batch_size=len(images), max_length=2048, doc_type="image",
        )
        return [scores] if isinstance(scores, (float, int)) else scores


DEFAULT_RERANKER = "Qwen3-VL-Reranker-8B"
RERANKERS = {
    DEFAULT_RERANKER: Qwen3VLReranker, 
    "jina-reranker-m0": JinaRerankerM0
}
