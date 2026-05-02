def _case(name, dataset_args, config_overrides):
    return {
        "name": name,
        "dataset_args": dataset_args,
        "config": config_overrides,
    }

CASES = [
    # Topic classification (multiclass)
    _case(
        name="dbpedia14_distilbert_base_uncased_seqcls",
        dataset_args={
            "dataset_name": "dbpedia_14",
            "train_split": "train",
            "test_split": "test",
            "text_column": "content",
            "label_column": "label",
            "max_samples": 1500,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),
    _case(
        name="yahoo_answers_topics_bert_base_uncased_seqcls",
        dataset_args={
            "dataset_name": "yahoo_answers_topics",
            "train_split": "train",
            "test_split": "test",
            # Yahoo has "question_title", "question_content", "best_answer".
            # If your loader supports only a single text column, you can start with "question_content".
            "text_column": "question_content",
            "label_column": "topic",
            "max_samples": 1500,
            "max_length": 128,
            "hf_model_id": "bert-base-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
        },
    ),

    _case(
        name="wnut17_bert_base_cased_token",
        dataset_args={
            "dataset_name": "wnut_17",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 600,
            "max_length": 128,
            "hf_model_id": "bert-base-cased",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),
    _case(
        name="wnut17_distilbert_token",
        dataset_args={
            "dataset_name": "wnut_17",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 500,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),
    _case(
        name="sst2_distilbert_base",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "sentence",
            "label_column": "label",
            "max_samples": 600,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),
    _case(
        name="sst2_distilbert_preft",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "sentence",
            "label_column": "label",
            "max_samples": 600,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        config_overrides={
            "learning_rate": 2e-5,
        },
    ),
    
    _case(
        name="ag_news_distilbert_base",
        dataset_args={
            "dataset_name": "ag_news",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),

    _case(
        name="conll2003_bert_base_cased_token",
        dataset_args={
            "dataset_name": "conll2003",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "bert-base-cased",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),

    _case(
        name="conll2003_distilbert_base_cased_token",
        dataset_args={
            "dataset_name": "conll2003",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "distilbert-base-cased",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),
    _case(
        name="conll2003_roberta_base_token",
        dataset_args={
            "dataset_name": "conll2003",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "roberta-base",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
        },
    ),

    _case(
        name="wnut17_distilbert_base_uncased_token",
        dataset_args={
            "dataset_name": "wnut_17",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 600,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),
    

# =========================================================
    # MULTI-LABEL CLASSIFICATION (needs loader support)
    # =========================================================
    _case(
        name="go_emotions_roberta_base_multilabel",
        dataset_args={
            "dataset_name": "go_emotions",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            # GoEmotions stores labels as a list under "labels"
            "label_column": "labels",
            "max_samples": 2000,
            "max_length": 128,
            "hf_model_id": "roberta-base",
            # Still sequence classification, but your adapter must treat labels as multi-hot.
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 2e-5,
        },
    ),


    _case(
        name="snli_roberta_base_nli",
        dataset_args={
            "dataset_name": "snli",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["premise", "hypothesis"],
            "label_column": "label",
            "max_samples": 1600,
            "max_length": 128,
            "hf_model_id": "roberta-base",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 2e-5,
        },
    ),

    # Paraphrase (pair classification). QQP is large; keep samples low.
    _case(
        name="qqp_minilm_l12_paraphrase",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "qqp",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["question1", "question2"],
            "label_column": "label",
            "max_samples": 2000,
            "max_length": 128,
            "hf_model_id": "microsoft/MiniLM-L12-H384-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
        },
    ),
    _case(
        name="mrpc_bert_base_uncased_paraphrase",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "mrpc",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "bert-base-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
        },
    ),

    # Toxicity / hate (binary)
    _case(
        name="tweet_eval_hate_roberta_base_seqcls",
        dataset_args={
            "dataset_name": "tweet_eval",
            "dataset_config": "hate",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 1500,
            "max_length": 128,
            "hf_model_id": "roberta-base",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 2e-5,
        },
    ),

    # Domain / Social media sentiment (pre-finetuned service-style model)
    _case(
        name="tweet_eval_sentiment_twitter_roberta_prefinetuned",
        dataset_args={
            "dataset_name": "tweet_eval",
            "dataset_config": "sentiment",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 1500,
            "max_length": 128,
            "hf_model_id": "cardiffnlp/twitter-roberta-base-sentiment",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 2e-5,
        },
    ),

    # POS tagging via Universal Dependencies (example: English EWT treebank)
    _case(
        name="ud_ewt_bert_base_cased_pos",
        dataset_args={
            "dataset_name": "universal_dependencies",
            "dataset_config": "en_ewt",
            "train_split": "train",
            "test_split": "validation",
            "tokens_column": "tokens",
            # UD typically provides "upos" (universal POS tags) and/or "xpos" (lang-specific).
            "label_column": "upos",
            "max_samples": 1200,
            "max_length": 128,
            "hf_model_id": "bert-base-cased",
            "hf_task": "token_classification",
        },
        config_overrides={
            "learning_rate": 5e-5,
        },
    ),

    # =========================================================
    # SEQUENCE CLASSIFICATION (binary / multiclass / NLI / paraphrase)
    # =========================================================

    # Sentiment (binary)
    _case(
        name="imdb_distilbert_base_uncased_seqcls",
        dataset_args={
            "dataset_name": "imdb",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 1200,
            "max_length": 256,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
        },
    ),
    _case(
        name="imdb_roberta_base_seqcls",
        dataset_args={
            "dataset_name": "imdb",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 1200,
            "max_length": 256,
            "hf_model_id": "roberta-base",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 2e-5,
        },
    ),

    _case(
        name="rotten_tomatoes_bert_base_uncased_seqcls",
        dataset_args={
            "dataset_name": "rotten_tomatoes",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 1200,
            "max_length": 128,
            "hf_model_id": "bert-base-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
        },
    ),

]

# ==========================================================
# Sentence Similarity & Pair Classification Case Set
# ==========================================================

SIMILARITY_CASES = [

    # ------------------------------------------------------
    # STS-B Regression – DistilBERT
    # ------------------------------------------------------
    _case(
        name="stsb_distilbert_sentence_similarity_regression",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "stsb",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "label_mode": "regression",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "sentence_similarity",
        },
        config_overrides={
            "learning_rate": 3e-5,
            "task_type": "regression",
        },
    ),

    # ------------------------------------------------------
    # STS-B Regression – BERT Base
    # ------------------------------------------------------
    _case(
        name="stsb_bert_base_sentence_similarity_regression",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "stsb",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "label_mode": "regression",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "bert-base-uncased",
            "hf_task": "sentence_similarity",
        },
        config_overrides={
            "learning_rate": 2e-5,
            "task_type": "regression",
        },
    ),

    # ------------------------------------------------------
    # STS-B Regression – RoBERTa Base
    # ------------------------------------------------------
    _case(
        name="stsb_roberta_base_sentence_similarity_regression",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "stsb",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "label_mode": "regression",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "roberta-base",
            "hf_task": "sentence_similarity",
        },
        config_overrides={
            "learning_rate": 2e-5,
            "task_type": "regression",
        },
    ),

    # ------------------------------------------------------
    # STS-B Regression – MiniLM (lightweight baseline)
    # ------------------------------------------------------
    _case(
        name="stsb_minilm_l12_sentence_similarity_regression",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "stsb",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "label_mode": "regression",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "microsoft/MiniLM-L12-H384-uncased",
            "hf_task": "sentence_similarity",
        },
        config_overrides={
            "learning_rate": 3e-5,
            "task_type": "regression",
        },
    ),

    # ------------------------------------------------------
    # MRPC – Sentence Pair Classification
    # ------------------------------------------------------
    _case(
        name="mrpc_distilbert_pair_classification",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "mrpc",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 3e-5,
            "task_type": "classification",
        },
    ),

    # ------------------------------------------------------
    # QQP – Paraphrase Detection
    # ------------------------------------------------------
    _case(
        name="qqp_roberta_base_pair_classification",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "qqp",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["question1", "question2"],
            "label_column": "label",
            "max_samples": 1000,
            "max_length": 128,
            "hf_model_id": "roberta-base",
            "hf_task": "sequence_classification",
        },
        config_overrides={
            "learning_rate": 2e-5,
            "task_type": "classification",
        },
    ),

    # ------------------------------------------------------
    # STS-B – Embedding Service (Inference Only)
    # ------------------------------------------------------
    _case(
        name="stsb_sentence_transformer_inference_only",
        dataset_args={
            "dataset_name": "glue",
            "dataset_config": "stsb",
            "train_split": "train",
            "test_split": "validation",
            "text_column": ["sentence1", "sentence2"],
            "label_column": "label",
            "label_mode": "regression",
            "max_samples": 800,
            "max_length": 128,
            "hf_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "hf_task": "sentence_similarity",
        },
        config_overrides={
            "task_type": "regression",
            "inference_only": True,
        },
    ),

]

FILLMASK_CASES = [
    _case(
        name="wikitext2_distilroberta_fill_mask",
        dataset_args={
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            "max_samples": 1200,
            "max_length": 128,
            "hf_model_id": "distilroberta-base",
            "hf_task": "fill_mask",
            "mlm_probability": 0.15,
            "label_pad_value": -100,
        },
        config_overrides={"learning_rate": 5e-5},
    ),

    _case(
        name="wikitext2_bert_base_masked_lm_alias",
        dataset_args={
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            "max_samples": 1200,
            "max_length": 128,
            "hf_model_id": "bert-base-uncased",
            "hf_task": "masked_lm",   # alias path
            "mlm_probability": 0.20,
            "label_pad_value": -100,
        },
        config_overrides={"learning_rate": 5e-5},
    ),

    _case(
        name="ag_news_distilbert_fill_mask_text_column",
        dataset_args={
            "dataset_name": "ag_news",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "max_samples": 1000,
            "max_length": 128,
            "hf_model_id": "distilbert-base-uncased",
            "hf_task": "fill_mask",
            "mlm_probability": 0.10,  # lower masking
            "label_pad_value": -100,
        },
        config_overrides={"learning_rate": 5e-5},
    ),

    _case(
        name="imdb_roberta_fill_mask_longer_context",
        dataset_args={
            "dataset_name": "imdb",
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "max_samples": 800,
            "max_length": 256,        # longer sequence behavior
            "hf_model_id": "roberta-base",
            "hf_task": "fill_mask",
            "mlm_probability": 0.15,
            "label_pad_value": -100,
        },
        config_overrides={"learning_rate": 3e-5},
    ),

    _case(
        name="tweet_eval_sentiment_fill_mask_short_text",
        dataset_args={
            "dataset_name": "tweet_eval",
            "dataset_config": "sentiment",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            "max_samples": 1000,
            "max_length": 64,         # short text stress
            "hf_model_id": "vinai/bertweet-base",
            "hf_task": "fill_mask",
            "mlm_probability": 0.25,  # higher masking
            "label_pad_value": -100,
        },
        config_overrides={"learning_rate": 3e-5},
    ),

    _case(
        name="dbpedia14_bert_fill_mask_mlm_alias",
        dataset_args={
            "dataset_name": "dbpedia_14",
            "train_split": "train",
            "test_split": "test",
            "text_column": "content",
            "max_samples": 1200,
            "max_length": 128,
            "hf_model_id": "bert-base-uncased",
            "hf_task": "mlm",         # alias path
            "mlm_probability": 0.15,
            "label_pad_value": -100,
        },
        config_overrides={"learning_rate": 5e-5},
    ),
]

COMPOSITION_BENCHMARKS = [
    {
        "name": "detector_to_captioner_to_classifier",
        "stages": ["object_detection", "image_captioning", "sequence_classification"],
        "report": {
            "primary_metric": "cider",
            "secondary_metric": "bleu",
            "stage_metrics": {
                "object_detection": ["map", "map@0.5"],
                "image_captioning": ["cider", "bleu"],
                "sequence_classification": ["accuracy", "f1"],
            },
        },
    },
    {
        "name": "retriever_to_vqa",
        "stages": ["text_image_retrieval", "visual_question_answering"],
        "report": {
            "primary_metric": "exact_match",
            "secondary_metric": "r@5",
            "stage_metrics": {
                "text_image_retrieval": ["r@1", "r@5", "r@10"],
                "visual_question_answering": ["exact_match"],
            },
        },
    },
]
