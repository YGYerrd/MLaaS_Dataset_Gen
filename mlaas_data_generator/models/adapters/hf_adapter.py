from .hf_core import HFCore
from .hf_cache import get_cached_model
from ...hf_tasks import resolve_model_hf_task
from .hf_task import (
    SequenceClassificationSpec,
    SentenceSimilaritySpec,
    TokenClassificationSpec,
    FillMaskSpec,
    CausalLMGenerationSpec,
    Seq2SeqGenerationSpec,
    ImageClassificationSpec,
    ObjectDetectionSpec,
    ImageSegmentationSpec,
    ImageCaptioningSpec,
    TextImageRetrievalSpec,
    VQASpec,
)


def resolve_hf_task(loader_template=None, hf_task=None):
    return resolve_model_hf_task(loader_template=loader_template, hf_task=hf_task)


def build_task_spec(hf_task=None, *, loader_template=None, num_labels=None, multilabel=False, label_format="single_index"):
    task = resolve_hf_task(loader_template=loader_template, hf_task=hf_task)
    if task == "token_classification":
        return task, TokenClassificationSpec(multilabel=multilabel, label_format=label_format)
    if task == "sentence_similarity":
        resolved_num_labels = None if num_labels is None else int(num_labels)
        is_regression = (str(label_format).lower() == "continuous") or (resolved_num_labels == 1)
        return task, SentenceSimilaritySpec(is_regression=is_regression)
    if task == "fill_mask":
        return task, FillMaskSpec()
    if task == "causal_lm_generation":
        return task, CausalLMGenerationSpec()
    if task == "seq2seq_generation":
        return task, Seq2SeqGenerationSpec()
    if task == "image_classification":
        return "image_classification", ImageClassificationSpec()
    if task == "image_detection":
        return "image_detection", ObjectDetectionSpec()
    if task == "image_segmentation":
        return "image_segmentation", ImageSegmentationSpec()
    if task == "image_captioning":
        return "image_captioning", ImageCaptioningSpec()
    if task == "text_image_retrieval":
        return "text_image_retrieval", TextImageRetrievalSpec()
    if task == "visual_question_answering":
        return "visual_question_answering", VQASpec(label_format=label_format)
    return "sequence_classification", SequenceClassificationSpec(multilabel=multilabel, label_format=label_format)


class TransformersTextFineTuneAdapter:
    def __init__(
        self,
        model_id,
        num_labels,
        max_length=128,
        batch_size=16,
        device=None,
        mixed_precision=None,
        precision_type="fp16",
        hf_task="sequence_classification",
        loader_template=None,
        label_pad_value=-100,
        multilabel=False,
        label_format="single_index",
        generation_config=None,
        task_tag=None,
    ):
        resolved_task, spec = build_task_spec(
            hf_task,
            loader_template=loader_template,
            num_labels=num_labels,
            multilabel=multilabel,
            label_format=label_format,
        )
        self.resolved_hf_task = resolved_task
        self.loader_template = loader_template
        self.core = HFCore(
            model_id=model_id,
            num_labels=(None if num_labels is None else int(num_labels)),
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            mixed_precision=mixed_precision,
            precision_type=precision_type,
            task_spec=spec,
            label_pad_value=int(label_pad_value),
            generation_config=generation_config,
            task_tag=task_tag,
        )
        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = self.core.device

    def count_params(self):
        return self.core.count_params()

    def get_weights(self):
        return self.core.get_weights()

    def set_weights(self, weights_dict):
        self.core.set_weights(weights_dict)

    def fit(
        self,
        x,
        y,
        epochs=1,
        lr=5e-5,
        optimizer="adamw",
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        max_train_time_s=60,
        progress_log_interval=None,
    ):
        return self.core.finetune(
            x,
            y,
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_train_time_s=max_train_time_s,
            progress_log_interval=progress_log_interval,
        )

    def evaluate(self, x, y, inference_only=False, max_eval_time_s=None, progress_log_interval=None):
        loss, primary, secondary, qos = self.core.eval(
            x,
            y,
            inference_only=inference_only,
            max_eval_time_s=max_eval_time_s,
            progress_log_interval=progress_log_interval,
        )
        return loss, primary, secondary, qos


class TransformersTextClassifierAdapter:
    def __init__(
        self,
        model_id,
        max_length=128,
        batch_size=16,
        device=None,
        mixed_precision=None,
        precision_type="fp16",
        hf_task="sequence_classification",
        loader_template=None,
        generation_config=None,
        task_tag=None,
    ):
        task, spec = build_task_spec(hf_task, loader_template=loader_template, num_labels=None)
        core = HFCore(
            model_id=model_id,
            num_labels=None,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            mixed_precision=mixed_precision,
            precision_type=precision_type,
            task_spec=spec,
            generation_config=generation_config,
            task_tag=task_tag,
        )

        transformers = core.transformers
        if core.model is None:
            def _load_model():
                if task == "causal_lm_generation":
                    return transformers.AutoModelForCausalLM.from_pretrained(model_id)
                if task == "seq2seq_generation":
                    return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
                if task == "fill_mask":
                    return transformers.AutoModelForMaskedLM.from_pretrained(model_id)
                if task == "token_classification":
                    return transformers.AutoModelForTokenClassification.from_pretrained(model_id)
                if task == "image_classification":
                    return transformers.AutoModelForImageClassification.from_pretrained(model_id)
                if task == "image_detection":
                    return transformers.AutoModelForObjectDetection.from_pretrained(model_id)
                if task == "image_segmentation":
                    return transformers.AutoModelForSemanticSegmentation.from_pretrained(model_id)
                if task == "image_captioning":
                    for auto_model_name in (
                        "AutoModelForVision2Seq",
                        "AutoModelForImageTextToText",
                        "BlipForConditionalGeneration",
                        "GitForCausalLM",
                        "AutoModelForCausalLM",
                    ):
                        AutoModel = getattr(transformers, auto_model_name, None)
                        if AutoModel is not None:
                            return AutoModel.from_pretrained(model_id)
                    raise AttributeError(
                        "transformers is missing image-captioning AutoModel/GIT/BLIP loaders"
                    )
                if task == "text_image_retrieval":
                    return transformers.AutoModel.from_pretrained(model_id)
                if task == "visual_question_answering":
                    for auto_model_name in (
                        "AutoModelForVisualQuestionAnswering",
                        "AutoModelForVision2Seq",
                        "AutoModelForImageTextToText",
                        "BlipForQuestionAnswering",
                        "GitForCausalLM",
                        "AutoModelForCausalLM",
                    ):
                        AutoModel = getattr(transformers, auto_model_name, None)
                        if AutoModel is not None:
                            try:
                                return AutoModel.from_pretrained(model_id)
                            except Exception:
                                continue
                    return transformers.AutoModelForVisualQuestionAnswering.from_pretrained(model_id)
                return transformers.AutoModelForSequenceClassification.from_pretrained(model_id)

            core.model, core.model_load_s, core.model_cache_hit = get_cached_model(
                hf_model_id=model_id,
                task=task,
                device=core.device,
                loader_fn=_load_model,
            )

        core.model.to(core.device)
        core._configure_memory_optimizations()
        core.model.eval()
        core.sync_effective_max_length()

        self.core = core
        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = self.core.device
        self.resolved_hf_task = task
        self.loader_template = loader_template

    def count_params(self):
        return self.core.count_params()

    def get_weights(self):
        return self.core.get_weights()

    def set_weights(self, weights_dict):
        self.core.set_weights(weights_dict)

    def evaluate(self, x, y, inference_only=True, max_eval_time_s=None, progress_log_interval=None):
        loss, primary, secondary, qos = self.core.eval(
            x,
            y,
            inference_only=inference_only,
            max_eval_time_s=max_eval_time_s,
            progress_log_interval=progress_log_interval,
        )

        qos = dict(qos)
        if "eval_latency_ms_mean" in qos:
            qos["inference_latency_ms_mean"] = qos.pop("eval_latency_ms_mean")
        if "eval_latency_ms_p95" in qos:
            qos["inference_latency_ms_p95"] = qos.pop("eval_latency_ms_p95")
        if "eval_latency_ms_steady_mean" in qos:
            qos["inference_latency_ms_steady_mean"] = qos.pop("eval_latency_ms_steady_mean")
        if "eval_latency_ms_steady_p95" in qos:
            qos["inference_latency_ms_steady_p95"] = qos.pop("eval_latency_ms_steady_p95")
        if "eval_throughput_eps" in qos:
            qos["throughput_eps"] = qos.pop("eval_throughput_eps")

        return loss, primary, secondary, qos
