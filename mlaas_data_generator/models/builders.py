from __future__ import annotations
from typing import Sequence

from .adapters.generic_adapters import KMeansAdapter, make_random_forest
from .label_schema import infer_ignore_index, infer_label_format, infer_num_labels


def _import_keras_stack():
    try:
        from keras import layers, models, optimizers, regularizers
        from keras.applications import MobileNetV2
        from keras.applications.mobilenet_v2 import preprocess_input
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(
            "Keras/TensorFlow model paths require 'tensorflow' and 'keras' in the active environment."
        ) from exc
    return layers, models, optimizers, regularizers, MobileNetV2, preprocess_input, tf


def _make_optimizer(name: str, lr: float):
    _, _, optimizers, _, _, _, _ = _import_keras_stack()
    if name is None or name.lower() == "none" or lr is None:
        return None
    name = name.lower()
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.0)
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    if name == "adagrad":
        return optimizers.Adagrad(learning_rate=lr)
    if name == "adamw":
        return optimizers.AdamW(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)

from .adapters.hf_adapter import resolve_hf_task

def create_model(
    input_shape,
    num_classes,
    hidden_layers: Sequence[int] = (64,),
    learning_rate: float = 0.01,
    activation: str = "relu",
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    task_type: str = "classification",
    model_type: str | None = None,
    meta=None,
    **kwargs
):
    # Allow meta to be passed via kwargs as well
    if meta is None:
        meta = kwargs.get("meta")

    model_choice = (model_type or "").lower()

    # ----------------------------
    # HF adapters (prefer loader meta)
    # ----------------------------
    if model_choice in ("hf_finetune", "hf_train", "transformers_finetune"):
        from .adapters.hf_adapter import TransformersTextFineTuneAdapter

        model_id = None
        if isinstance(meta, dict):
            model_id = meta.get("hf_model_id")
        model_id = model_id or kwargs.get("hf_model_id") or kwargs.get("model_id") or kwargs.get("model_name")
        if not model_id:
            raise ValueError("HF fine-tune model_type requires hf_model_id=<huggingface_model_repo_id>")

        loader_template = (meta or {}).get("loader_template") if isinstance(meta, dict) else None
        loader_template = loader_template or kwargs.get("loader_template")
        hf_task = None
        if isinstance(meta, dict):
            hf_task = meta.get("hf_task")
        hf_task = resolve_hf_task(loader_template=loader_template, hf_task=hf_task or kwargs.get("hf_task", "sequence_classification"))

        label_pad_value = infer_ignore_index(meta if isinstance(meta, dict) else None, default=int(kwargs.get("label_pad_value", -100)))
        
        # max_length: prefer explicit, else meta input_shape, else fallback
        max_length = kwargs.get("max_length", None)
        if max_length is None and isinstance(meta, dict):
            ish = meta.get("input_shape")
            if ish and len(ish) >= 1:
                max_length = ish[0]
        if max_length is None:
            # fall back to input_shape only if provided
            if input_shape is not None and len(input_shape) >= 1:
                max_length = input_shape[0]
            else:
                max_length = 128
        max_length = int(max_length)

        batch_size = int(kwargs.get("batch_size", 16))
        device = kwargs.get("device", None)

        label_format = infer_label_format(meta if isinstance(meta, dict) else None, task_type=task_type)
        multilabel = label_format in {"multilabel", "multihot"}
        multilabel = bool(kwargs.get("multilabel", multilabel))
        resolved_num_labels = infer_num_labels(meta if isinstance(meta, dict) else None, fallback=num_classes)
        if hf_task in {"fill_mask", "causal_lm_generation", "seq2seq_generation"}:
            resolved_num_labels = None

        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", (meta or {}).get("max_new_tokens") if isinstance(meta, dict) else None),
            "num_beams": kwargs.get("num_beams", (meta or {}).get("num_beams") if isinstance(meta, dict) else None),
            "do_sample": kwargs.get("do_sample", (meta or {}).get("do_sample") if isinstance(meta, dict) else None),
            "temperature": kwargs.get("temperature", (meta or {}).get("temperature") if isinstance(meta, dict) else None),
            "top_k": kwargs.get("top_k", (meta or {}).get("top_k") if isinstance(meta, dict) else None),
            "top_p": kwargs.get("top_p", (meta or {}).get("top_p") if isinstance(meta, dict) else None),
            "length_penalty": kwargs.get("length_penalty", (meta or {}).get("length_penalty") if isinstance(meta, dict) else None),
        }

        return TransformersTextFineTuneAdapter(
            model_id=model_id,
            num_labels=(None if resolved_num_labels is None else int(resolved_num_labels)),
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            mixed_precision=kwargs.get("mixed_precision"),
            hf_task=hf_task,
            loader_template=loader_template,
            label_pad_value=label_pad_value,
            multilabel=multilabel,
            label_format=label_format,
            generation_config=generation_config,
            task_tag=kwargs.get("task_tag", (meta or {}).get("task_tag") if isinstance(meta, dict) else None),
        )

    if model_choice in ("hf", "hf_text", "transformers"):
        from .adapters.hf_adapter import TransformersTextClassifierAdapter

        model_id = None
        if isinstance(meta, dict):
            model_id = meta.get("hf_model_id")
        model_id = model_id or kwargs.get("hf_model_id") or kwargs.get("model_id") or kwargs.get("model_name")
        if not model_id:
            raise ValueError("HF model_type requires hf_model_id=<huggingface_model_repo_id>")

        loader_template = (meta or {}).get("loader_template") if isinstance(meta, dict) else None
        loader_template = loader_template or kwargs.get("loader_template")
        hf_task = None
        if isinstance(meta, dict):
            hf_task = meta.get("hf_task")
        hf_task = resolve_hf_task(loader_template=loader_template, hf_task=hf_task or kwargs.get("hf_task", "sequence_classification"))

        max_length = kwargs.get("max_length", None)
        if max_length is None and isinstance(meta, dict):
            ish = meta.get("input_shape")
            if ish and len(ish) >= 1:
                max_length = ish[0]
        if max_length is None:
            if input_shape is not None and len(input_shape) >= 1:
                max_length = input_shape[0]
            else:
                max_length = 128
        max_length = int(max_length)

        batch_size = int(kwargs.get("batch_size", 16))
        device = kwargs.get("device", None)

        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", (meta or {}).get("max_new_tokens") if isinstance(meta, dict) else None),
            "num_beams": kwargs.get("num_beams", (meta or {}).get("num_beams") if isinstance(meta, dict) else None),
            "do_sample": kwargs.get("do_sample", (meta or {}).get("do_sample") if isinstance(meta, dict) else None),
            "temperature": kwargs.get("temperature", (meta or {}).get("temperature") if isinstance(meta, dict) else None),
            "top_k": kwargs.get("top_k", (meta or {}).get("top_k") if isinstance(meta, dict) else None),
            "top_p": kwargs.get("top_p", (meta or {}).get("top_p") if isinstance(meta, dict) else None),
            "length_penalty": kwargs.get("length_penalty", (meta or {}).get("length_penalty") if isinstance(meta, dict) else None),
        }

        return TransformersTextClassifierAdapter(
            model_id=model_id,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            mixed_precision=kwargs.get("mixed_precision"),
            hf_task=hf_task,
            loader_template=loader_template,
            generation_config=generation_config,
            task_tag=kwargs.get("task_tag", (meta or {}).get("task_tag") if isinstance(meta, dict) else None),
        )

    # ----------------------------
    # Non-HF models (need rank)
    # ----------------------------
    if input_shape is None:
        raise ValueError("input_shape is required for non-HF models")
    rank = len(input_shape)
    if not model_choice:
        model_choice = ("cnn" if rank == 3 else "mlp")

    if task_type == "clustering":
        k_value = kwargs.get("k")
        if k_value is None:
            k_value = kwargs.get("clustering_k")
        if k_value is None:
            k = 3
        else:
            k = int(k_value)
            if k <= 0:
                raise ValueError("clustering_k must be a positive integer")
        init = kwargs.get("clustering_init", "k-means++")
        n_init = int(kwargs.get("clustering_n_init", 10))
        max_iter = int(kwargs.get("clustering_max_iter", 300))
        tol = float(kwargs.get("clustering_tol", 1e-4))
        seed = kwargs.get("random_state", kwargs.get("seed", None))
        return KMeansAdapter(
            input_shape=input_shape,
            k=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
        )

    is_regression = (task_type == "regression")
    label_format = infer_label_format(meta if isinstance(meta, dict) else None, task_type=task_type)
    resolved_num_labels = infer_num_labels(meta if isinstance(meta, dict) else None, fallback=num_classes)

    if is_regression:
        out_units = 1
        out_activation = "linear"
        loss = "mse"
        metrics = ["mse"]
    elif label_format in {"multilabel", "multihot"}:
        out_units = int(resolved_num_labels)
        out_activation = "sigmoid"
        loss = "binary_crossentropy"
        metrics = ["binary_accuracy"]
    elif label_format == "onehot":
        out_units = int(resolved_num_labels)
        out_activation = "softmax"
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        out_units = int(resolved_num_labels)
        out_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    if model_choice == "randomforest":
        rf_kwargs = {
            "n_estimators": int(kwargs.get("rf_trees", kwargs.get("n_estimators", 100))),
            "max_depth": kwargs.get("rf_max_depth", kwargs.get("max_depth", None)),
            "random_state": kwargs.get("seed", kwargs.get("random_state", None)),
        }
        return make_random_forest(task_type="regression" if is_regression else "classification", **rf_kwargs)

    layers, models, _, regularizers, MobileNetV2, preprocess_input, tf = _import_keras_stack()
    l2 = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None

    if rank == 3:
        if model_choice == "mobilenetv2":
            base = MobileNetV2(include_top=False, weights="imagenet", pooling=None)
            base.trainable = bool(kwargs.get("mobilenet_trainable", False))

            inputs = layers.Input(shape=input_shape)
            x = inputs
            if input_shape[-1] == 1:
                x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
            elif input_shape[-1] != 3:
                raise ValueError(f"MobileNetV2 expects 1 or 3 channel input; got shape {input_shape}")
            x = layers.Resizing(96, 96)(x)
            x = layers.Lambda(lambda t: tf.cast(t, tf.float32))(x)
            x = layers.Lambda(preprocess_input)(x)
            x = base(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
            outputs = layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2)(x)
            model = models.Model(inputs=inputs, outputs=outputs, name="mlaas_mobilenetv2")
        else:
            model = models.Sequential(name="mlaas_cnn")
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Conv2D(32, 3, padding="same", activation=activation, kernel_regularizer=l2))
            model.add(layers.MaxPooling2D())
            model.add(layers.Conv2D(64, 3, padding="same", activation=activation, kernel_regularizer=l2))
            model.add(layers.MaxPooling2D())
            model.add(layers.Flatten())
            for units in hidden_layers:
                model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2))
                if dropout > 0:
                    model.add(layers.Dropout(dropout))
            model.add(layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2))

    elif rank == 1:
        model_name = "mlaas_logreg" if model_choice == "logreg" else "mlaas_mlp"
        model = models.Sequential(name=model_name)

        model.add(layers.Input(shape=input_shape))
        layers_to_use = [] if model_choice == "logreg" else list(hidden_layers)
        for units in layers_to_use:
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2))
            if dropout and dropout > 0:
                model.add(layers.Dropout(dropout))
        model.add(layers.Dense(out_units, activation=out_activation, kernel_regularizer=l2))

    else:
        raise ValueError(f"Unsupported input_shape {input_shape}; rank {rank} not handled.")

    opt = _make_optimizer(optimizer, learning_rate)
    if opt is not None:
        model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model
