SELECT *
FROM (
    WITH
    m AS (
        SELECT
            service_id,
            metric_name,
            COALESCE(value_num, CAST(value_int AS REAL), CAST(value_bool AS REAL)) AS num_value,
            COALESCE(value_text, CAST(value_num AS TEXT), CAST(value_int AS TEXT), CAST(value_bool AS TEXT), value_json) AS text_value
        FROM service_metrics
    ),

    metric_values AS (
        SELECT
            service_id,
            MAX(CASE WHEN metric_name = 'primary_metric_name' THEN lower(text_value) END) AS primary_metric_name,
            MAX(CASE WHEN metric_name = 'auxiliary_metric_name' THEN lower(text_value) END) AS auxiliary_metric_name,

            MAX(CASE WHEN metric_name = 'metric_score' THEN num_value END) AS metric_score,
            MAX(CASE WHEN metric_name = 'accuracy' THEN num_value END) AS accuracy,
            MAX(CASE WHEN metric_name = 'f1' THEN num_value END) AS f1,
            MAX(CASE WHEN metric_name IN ('map', 'mAP') THEN num_value END) AS map_value,
            MAX(CASE WHEN metric_name = 'map@0.5' THEN num_value END) AS map_at_50,
            MAX(CASE WHEN metric_name = 'map@0.75' THEN num_value END) AS map_at_75,
            MAX(CASE WHEN metric_name = 'iou' THEN num_value END) AS iou,
            MAX(CASE WHEN metric_name = 'dice' THEN num_value END) AS dice,
            MAX(CASE WHEN metric_name = 'pixel_accuracy' THEN num_value END) AS pixel_accuracy,
            MAX(CASE WHEN metric_name = 'masked_accuracy' THEN num_value END) AS masked_accuracy,
            MAX(CASE WHEN metric_name = 'token_accuracy' THEN num_value END) AS token_accuracy,
            MAX(CASE WHEN metric_name = 'r@1' THEN num_value END) AS recall_at_1,
            MAX(CASE WHEN metric_name = 'r@5' THEN num_value END) AS recall_at_5,
            MAX(CASE WHEN metric_name = 'silhouette' THEN num_value END) AS silhouette,
            MAX(CASE WHEN metric_name = 'inertia' THEN num_value END) AS inertia,
            MAX(CASE WHEN metric_name = 'rouge1' THEN num_value END) AS rouge1,
            MAX(CASE WHEN metric_name = 'rouge2' THEN num_value END) AS rouge2,
            MAX(CASE WHEN metric_name IN ('rougel', 'rouge-l', 'rouge_l') THEN num_value END) AS rougel,
            MAX(CASE WHEN metric_name IN ('bleu', 'sacrebleu') THEN num_value END) AS bleu,
            MAX(CASE WHEN metric_name = 'cider' THEN num_value END) AS cider,
            MAX(CASE WHEN metric_name = 'exact_match' THEN num_value END) AS exact_match,
            MAX(CASE WHEN metric_name = 'rmse' THEN num_value END) AS rmse,
            MAX(CASE WHEN metric_name = 'mae' THEN num_value END) AS mae,
            MAX(CASE WHEN metric_name = 'pearson' THEN num_value END) AS pearson,
            MAX(CASE WHEN metric_name = 'spearman' THEN num_value END) AS spearman,
            MAX(CASE WHEN metric_name IN ('loss', 'train_loss', 'cross_entropy_loss') THEN num_value END) AS loss,
            MAX(CASE WHEN metric_name IN ('perplexity', 'perplexity_proxy') THEN num_value END) AS perplexity,

            MAX(CASE WHEN metric_name IN ('latency', 'inference_latency_s', 'inference_latency_s_mean') THEN num_value END) AS latency,
            MAX(CASE WHEN metric_name IN ('tail_latency', 'inference_latency_s_p95', 'latency_s_p95') THEN num_value END) AS tail_latency,
            MAX(CASE WHEN metric_name = 'participation_rate' THEN num_value END) AS participation_rate,
            MAX(CASE WHEN metric_name = 'client_participation_rate' THEN num_value END) AS client_participation_rate,
            MAX(CASE WHEN metric_name = 'reliability_score' THEN num_value END) AS reliability_score,
            MAX(CASE WHEN metric_name = 'compute_time_s' THEN num_value END) AS mean_compute_time,
            MAX(CASE WHEN metric_name = 'raw_resource_cost' THEN num_value END) AS raw_resource_cost,
            MAX(CASE WHEN metric_name = 'resource_cost_score' THEN num_value END) AS stored_resource_cost_score,

            MAX(CASE WHEN metric_name IN ('model_size', 'params_count', 'model_params_count') THEN num_value END) AS model_size,
            MAX(CASE WHEN metric_name = 'downloads' THEN num_value END) AS downloads,
            MAX(CASE WHEN metric_name = 'hf_model_downloads' THEN num_value END) AS hf_model_downloads,
            MAX(CASE WHEN metric_name = 'hf_dataset_downloads' THEN num_value END) AS hf_dataset_downloads,
            MAX(CASE WHEN metric_name = 'likes' THEN num_value END) AS likes,
            MAX(CASE WHEN metric_name = 'hf_model_likes' THEN num_value END) AS hf_model_likes,
            MAX(CASE WHEN metric_name = 'hf_dataset_likes' THEN num_value END) AS hf_dataset_likes,
            MAX(CASE WHEN metric_name = 'learning_rate' THEN num_value END) AS learning_rate,
            MAX(CASE WHEN metric_name = 'batch_size' THEN num_value END) AS batch_size,
            MAX(CASE WHEN metric_name IN ('data_distribution', 'split_strategy') THEN text_value END) AS data_distribution,
            MAX(CASE WHEN metric_name = 'split_strategy_requested' THEN text_value END) AS split_strategy_requested,
            MAX(CASE WHEN metric_name = 'split_strategy_effective' THEN text_value END) AS split_strategy_effective,
            MAX(CASE WHEN metric_name = 'split_skew_axis' THEN text_value END) AS split_skew_axis,
            MAX(CASE WHEN metric_name = 'split_skew_axis_effective' THEN text_value END) AS split_skew_axis_effective,
            MAX(CASE WHEN metric_name = 'split_bucket_spec_json' THEN text_value END) AS split_bucket_spec_json,
            MAX(CASE WHEN metric_name IN ('dataset_distribution_json', 'split_provenance_json') THEN text_value END) AS dataset_distribution_json,
            MAX(CASE WHEN metric_name IN ('dataset_size', 'train_set_size') THEN num_value END) AS dataset_size,
            MAX(CASE WHEN metric_name = 'explainability_score' THEN num_value END) AS explainability_score,
            MAX(CASE WHEN metric_name = 'hf_model_id' THEN text_value END) AS hf_model_id,
            MAX(CASE WHEN metric_name = 'hf_dataset_id' THEN text_value END) AS hf_dataset_id,
            MAX(CASE WHEN metric_name = 'update_signature_id' THEN text_value END) AS update_signature_id,
            MAX(CASE WHEN metric_name = 'signature_dim' THEN num_value END) AS signature_dim,
            MAX(CASE WHEN metric_name = 'signature_norm' THEN num_value END) AS signature_norm,
            MAX(CASE WHEN metric_name = 'update_signature_path' THEN text_value END) AS update_signature_path,
            MAX(CASE WHEN metric_name = 'update_signature_method' THEN text_value END) AS update_signature_method
        FROM m
        GROUP BY service_id
    ),

    split_summary AS (
        SELECT
            service_id,
            MAX(CASE WHEN split_name = 'train' THEN samples_count END) AS train_samples_count,
            MAX(CASE WHEN split_name = 'benchmark' THEN samples_count END) AS benchmark_samples_count,
            MAX(CASE WHEN split_name = 'train' THEN data_distribution_json END) AS train_distribution_json,
            MAX(CASE WHEN split_name = 'benchmark' THEN data_distribution_json END) AS benchmark_distribution_json,
            MAX(
                CASE
                    WHEN split_name = 'train'
                    THEN COALESCE(
                        json_extract(split_config_json, '$.requested_strategy'),
                        json_extract(split_config_json, '$.effective_strategy'),
                        json_extract(split_config_json, '$.distribution_type'),
                        json_extract(split_config_json, '$.strategy')
                    )
                END
            ) AS train_distribution_name,
            MAX(CASE WHEN split_name = 'train' THEN json_extract(split_config_json, '$.requested_strategy') END) AS train_requested_strategy,
            MAX(CASE WHEN split_name = 'train' THEN json_extract(split_config_json, '$.effective_strategy') END) AS train_effective_strategy,
            MAX(CASE WHEN split_name = 'train' THEN json_extract(split_config_json, '$.requested_axis') END) AS train_requested_axis,
            MAX(CASE WHEN split_name = 'train' THEN json_extract(split_config_json, '$.effective_axis') END) AS train_effective_axis,
            MAX(CASE WHEN split_name = 'train' THEN json_extract(split_config_json, '$.bucket_spec') END) AS train_bucket_spec,
            group_concat(
                split_name || ': ' || COALESCE(data_distribution_json, '{}'),
                char(10)
            ) AS dataset_distributions
        FROM (
            SELECT *
            FROM service_split_provenance
            ORDER BY service_id, split_name
        )
        GROUP BY service_id
    ),

    failure_counts AS (
        SELECT
            service_id,
            COUNT(*) AS historical_failure_count
        FROM service_failures
        WHERE service_id IS NOT NULL
        GROUP BY service_id
    ),

    latest_failure_ids AS (
        SELECT
            service_id,
            MAX(failure_id) AS latest_failure_id
        FROM service_failures
        WHERE service_id IS NOT NULL
        GROUP BY service_id
    ),

    latest_failures AS (
        SELECT
            sf.service_id,
            sf.failure_stage,
            sf.error_message
        FROM service_failures sf
        JOIN latest_failure_ids lf
          ON lf.service_id = sf.service_id
         AND lf.latest_failure_id = sf.failure_id
    ),

    raw_base AS (
        SELECT
            s.service_id AS run_id,
            s.status,
            s.created_at,
            COALESCE(s.task_family, s.task_type) AS task_family,
            COALESCE(s.task_type, s.task_family) AS task_type,
            s.hf_task,
            COALESCE(s.dataset_name, s.dataset) AS dataset,
            COALESCE(s.model_type, s.model_id) AS model_type,
            COALESCE(mv.hf_model_id, s.model_id) AS hf_model_id,
            COALESCE(mv.hf_dataset_id, s.dataset_name, s.dataset) AS hf_dataset_id,

            COALESCE(
                mv.primary_metric_name,
                CASE
                    WHEN mv.accuracy IS NOT NULL THEN 'accuracy'
                    WHEN mv.f1 IS NOT NULL THEN 'f1'
                    WHEN mv.map_value IS NOT NULL THEN 'map'
                    WHEN mv.iou IS NOT NULL THEN 'iou'
                    WHEN mv.masked_accuracy IS NOT NULL THEN 'masked_accuracy'
                    WHEN mv.recall_at_1 IS NOT NULL THEN 'r@1'
                    WHEN mv.token_accuracy IS NOT NULL THEN 'token_accuracy'
                    WHEN mv.pixel_accuracy IS NOT NULL THEN 'pixel_accuracy'
                    WHEN mv.silhouette IS NOT NULL THEN 'silhouette'
                    WHEN mv.rouge1 IS NOT NULL THEN 'rouge1'
                    WHEN mv.bleu IS NOT NULL THEN 'bleu'
                    WHEN mv.cider IS NOT NULL THEN 'cider'
                    WHEN mv.exact_match IS NOT NULL THEN 'exact_match'
                    WHEN mv.pearson IS NOT NULL THEN 'pearson'
                    WHEN mv.rmse IS NOT NULL THEN 'rmse'
                    WHEN mv.loss IS NOT NULL THEN 'loss'
                    WHEN mv.perplexity IS NOT NULL THEN 'perplexity'
                    WHEN mv.metric_score IS NOT NULL THEN 'metric_score'
                END
            ) AS primary_metric_name,

            COALESCE(
                mv.auxiliary_metric_name,
                CASE
                    WHEN mv.f1 IS NOT NULL THEN 'f1'
                    WHEN mv.map_at_50 IS NOT NULL THEN 'map@0.5'
                    WHEN mv.dice IS NOT NULL THEN 'dice'
                    WHEN mv.recall_at_5 IS NOT NULL THEN 'r@5'
                    WHEN mv.rouge2 IS NOT NULL THEN 'rouge2'
                    WHEN mv.spearman IS NOT NULL THEN 'spearman'
                    WHEN mv.mae IS NOT NULL THEN 'mae'
                    WHEN mv.perplexity IS NOT NULL THEN 'perplexity'
                    WHEN mv.inertia IS NOT NULL THEN 'inertia'
                    WHEN mv.loss IS NOT NULL THEN 'loss'
                END
            ) AS auxiliary_metric_name,

            mv.metric_score,
            mv.accuracy,
            mv.f1,
            mv.map_value,
            mv.map_at_50,
            mv.map_at_75,
            mv.iou,
            mv.dice,
            mv.pixel_accuracy,
            mv.masked_accuracy,
            mv.token_accuracy,
            mv.recall_at_1,
            mv.recall_at_5,
            mv.silhouette,
            mv.inertia,
            mv.rouge1,
            mv.rouge2,
            mv.rougel,
            mv.bleu,
            mv.cider,
            mv.exact_match,
            mv.rmse,
            mv.mae,
            mv.pearson,
            mv.spearman,
            mv.loss,
            mv.perplexity,

            mv.latency,
            COALESCE(mv.tail_latency, mv.latency) AS tail_latency,
            COALESCE(mv.participation_rate, mv.client_participation_rate) AS participation_rate,
            mv.reliability_score,
            mv.mean_compute_time,
            mv.raw_resource_cost,
            mv.stored_resource_cost_score,
            mv.model_size,
            COALESCE(mv.downloads, mv.hf_model_downloads, mv.hf_dataset_downloads) AS downloads,
            COALESCE(mv.likes, mv.hf_model_likes, mv.hf_dataset_likes) AS likes,
            mv.hf_model_downloads,
            mv.hf_dataset_downloads,
            mv.hf_model_likes,
            mv.hf_dataset_likes,
            CASE
                WHEN lower(COALESCE(s.training_regime, '')) IN ('inference_only', 'inference') THEN NULL
                ELSE mv.learning_rate
            END AS learning_rate,
            mv.batch_size,
            COALESCE(mv.split_strategy_requested, ss.train_requested_strategy) AS split_strategy_requested,
            COALESCE(mv.split_strategy_effective, ss.train_effective_strategy) AS split_strategy_effective,
            COALESCE(mv.split_skew_axis, ss.train_requested_axis) AS split_skew_axis,
            COALESCE(mv.split_skew_axis_effective, ss.train_effective_axis) AS split_skew_axis_effective,
            COALESCE(mv.split_bucket_spec_json, ss.train_bucket_spec) AS split_bucket_spec_json,
            COALESCE(mv.data_distribution, ss.train_distribution_name) AS data_distribution,
            COALESCE(ss.dataset_distributions, mv.dataset_distribution_json) AS dataset_distributions,
            COALESCE(mv.dataset_size, ss.train_samples_count, ss.benchmark_samples_count) AS dataset_size,
            mv.explainability_score,
            mv.update_signature_id,
            mv.signature_dim,
            mv.signature_norm,
            mv.update_signature_path,
            mv.update_signature_method,
            COALESCE(fc.historical_failure_count, 0) AS historical_failure_count,
            lf.failure_stage AS latest_failure_stage,
            lf.error_message AS latest_failure_message
        FROM services s
        LEFT JOIN metric_values mv
          ON mv.service_id = s.service_id
        LEFT JOIN split_summary ss
          ON ss.service_id = s.service_id
        LEFT JOIN failure_counts fc
          ON fc.service_id = s.service_id
        LEFT JOIN latest_failures lf
          ON lf.service_id = s.service_id
    ),

    raw AS (
        SELECT
            *,
            CASE replace(primary_metric_name, '-', '_')
                WHEN 'accuracy' THEN COALESCE(accuracy, metric_score)
                WHEN 'top1_accuracy' THEN COALESCE(accuracy, metric_score)
                WHEN 'f1' THEN COALESCE(f1, metric_score)
                WHEN 'macro_f1' THEN COALESCE(f1, metric_score)
                WHEN 'entity_f1' THEN COALESCE(f1, metric_score)
                WHEN 'map' THEN COALESCE(map_value, metric_score)
                WHEN 'mean_average_precision' THEN COALESCE(map_value, metric_score)
                WHEN 'map@0.5' THEN map_at_50
                WHEN 'map@0.75' THEN map_at_75
                WHEN 'iou' THEN COALESCE(iou, metric_score)
                WHEN 'miou' THEN COALESCE(iou, metric_score)
                WHEN 'mean_iou' THEN COALESCE(iou, metric_score)
                WHEN 'dice' THEN dice
                WHEN 'pixel_accuracy' THEN pixel_accuracy
                WHEN 'masked_accuracy' THEN COALESCE(masked_accuracy, metric_score)
                WHEN 'token_accuracy' THEN token_accuracy
                WHEN 'r@1' THEN COALESCE(recall_at_1, metric_score)
                WHEN 'silhouette' THEN COALESCE(silhouette, metric_score)
                WHEN 'rouge1' THEN COALESCE(rouge1, metric_score)
                WHEN 'rouge2' THEN rouge2
                WHEN 'rougel' THEN COALESCE(rougel, metric_score)
                WHEN 'rouge_l' THEN COALESCE(rougel, metric_score)
                WHEN 'bleu' THEN COALESCE(bleu, metric_score)
                WHEN 'sacrebleu' THEN COALESCE(bleu, metric_score)
                WHEN 'cider' THEN COALESCE(cider, metric_score)
                WHEN 'exact_match' THEN COALESCE(exact_match, metric_score)
                WHEN 'pearson' THEN COALESCE(pearson, metric_score)
                WHEN 'spearman' THEN COALESCE(spearman, metric_score)
                WHEN 'rmse' THEN rmse
                WHEN 'mae' THEN mae
                WHEN 'loss' THEN loss
                WHEN 'cross_entropy_loss' THEN loss
                WHEN 'perplexity' THEN perplexity
                WHEN 'perplexity_proxy' THEN perplexity
                WHEN 'metric_score' THEN metric_score
                ELSE COALESCE(
                    metric_score,
                    accuracy,
                    f1,
                    map_value,
                    iou,
                    pearson,
                    rmse,
                    loss
                )
            END AS primary_metric,

            CASE replace(auxiliary_metric_name, '-', '_')
                WHEN 'accuracy' THEN COALESCE(accuracy, metric_score)
                WHEN 'top1_accuracy' THEN COALESCE(accuracy, metric_score)
                WHEN 'f1' THEN COALESCE(f1, metric_score)
                WHEN 'macro_f1' THEN COALESCE(f1, metric_score)
                WHEN 'entity_f1' THEN COALESCE(f1, metric_score)
                WHEN 'map' THEN COALESCE(map_value, metric_score)
                WHEN 'mean_average_precision' THEN COALESCE(map_value, metric_score)
                WHEN 'map@0.5' THEN map_at_50
                WHEN 'map@0.75' THEN map_at_75
                WHEN 'iou' THEN COALESCE(iou, metric_score)
                WHEN 'miou' THEN COALESCE(iou, metric_score)
                WHEN 'mean_iou' THEN COALESCE(iou, metric_score)
                WHEN 'dice' THEN dice
                WHEN 'pixel_accuracy' THEN pixel_accuracy
                WHEN 'masked_accuracy' THEN COALESCE(masked_accuracy, metric_score)
                WHEN 'token_accuracy' THEN token_accuracy
                WHEN 'r@5' THEN recall_at_5
                WHEN 'silhouette' THEN COALESCE(silhouette, metric_score)
                WHEN 'inertia' THEN inertia
                WHEN 'rouge1' THEN COALESCE(rouge1, metric_score)
                WHEN 'rouge2' THEN rouge2
                WHEN 'rougel' THEN COALESCE(rougel, metric_score)
                WHEN 'rouge_l' THEN COALESCE(rougel, metric_score)
                WHEN 'bleu' THEN COALESCE(bleu, metric_score)
                WHEN 'sacrebleu' THEN COALESCE(bleu, metric_score)
                WHEN 'cider' THEN COALESCE(cider, metric_score)
                WHEN 'exact_match' THEN COALESCE(exact_match, metric_score)
                WHEN 'pearson' THEN COALESCE(pearson, metric_score)
                WHEN 'spearman' THEN COALESCE(spearman, metric_score)
                WHEN 'rmse' THEN rmse
                WHEN 'mae' THEN mae
                WHEN 'loss' THEN loss
                WHEN 'cross_entropy_loss' THEN loss
                WHEN 'perplexity' THEN perplexity
                WHEN 'perplexity_proxy' THEN perplexity
                WHEN 'auxiliary_metric' THEN NULL
                ELSE COALESCE(
                    f1,
                    mae,
                    spearman,
                    perplexity,
                    loss,
                    map_at_50,
                    dice,
                    recall_at_5,
                    inertia
                )
            END AS auxiliary_metric
        FROM raw_base
    ),

    scored AS (
        SELECT
            *,
            CASE
                WHEN replace(primary_metric_name, '-', '_') IN ('rmse', 'mae', 'loss', 'cross_entropy_loss', 'perplexity', 'perplexity_proxy')
                 AND primary_metric IS NOT NULL
                THEN 1.0 / (1.0 + primary_metric)
                ELSE primary_metric
            END AS primary_metric_higher_better,
            CASE
                WHEN replace(auxiliary_metric_name, '-', '_') IN ('rmse', 'mae', 'loss', 'cross_entropy_loss', 'perplexity', 'perplexity_proxy', 'inertia')
                 AND auxiliary_metric IS NOT NULL
                THEN 1.0 / (1.0 + auxiliary_metric)
                ELSE auxiliary_metric
            END AS auxiliary_metric_higher_better,
            CASE
                WHEN replace(primary_metric_name, '-', '_') = 'masked_accuracy' AND auxiliary_metric IS NULL THEN 1
                WHEN replace(primary_metric_name, '-', '_') = 'map' AND map_at_50 IS NULL THEN 1
                WHEN replace(primary_metric_name, '-', '_') = 'iou' AND dice IS NULL THEN 1
                WHEN replace(primary_metric_name, '-', '_') = 'loss'
                 AND replace(COALESCE(hf_task, ''), '-', '_') = 'causal_lm_generation'
                 AND perplexity IS NULL THEN 1
                ELSE 0
            END AS required_auxiliary_missing_flag,
            CASE
                WHEN replace(primary_metric_name, '-', '_') = 'masked_accuracy' AND COALESCE(primary_metric, metric_score) = 0.0 THEN 1
                WHEN replace(primary_metric_name, '-', '_') = 'rouge1' AND COALESCE(primary_metric, metric_score) = 0.0 THEN 1
                WHEN replace(primary_metric_name, '-', '_') = 'exact_match' AND COALESCE(primary_metric, metric_score) = 0.0 THEN 1
                ELSE 0
            END AS degenerate_zero_flag
        FROM raw
    ),

    final_values AS (
        SELECT
            *,
            CASE
                WHEN primary_metric_higher_better IS NULL THEN NULL
                WHEN MAX(primary_metric_higher_better) OVER () = MIN(primary_metric_higher_better) OVER () THEN 1.0
                ELSE max(0.0, min(1.0,
                    (primary_metric_higher_better - MIN(primary_metric_higher_better) OVER ())
                    / NULLIF(MAX(primary_metric_higher_better) OVER () - MIN(primary_metric_higher_better) OVER (), 0.0)
                ))
            END AS normalised_primary_metric,
            CASE
                WHEN auxiliary_metric_higher_better IS NULL THEN NULL
                WHEN MAX(auxiliary_metric_higher_better) OVER () = MIN(auxiliary_metric_higher_better) OVER () THEN 1.0
                ELSE max(0.0, min(1.0,
                    (auxiliary_metric_higher_better - MIN(auxiliary_metric_higher_better) OVER ())
                    / NULLIF(MAX(auxiliary_metric_higher_better) OVER () - MIN(auxiliary_metric_higher_better) OVER (), 0.0)
                ))
            END AS normalised_auxiliary_metric,
            CASE
                WHEN raw_resource_cost IS NULL THEN stored_resource_cost_score
                WHEN MAX(raw_resource_cost) OVER () = MIN(raw_resource_cost) OVER () THEN 1.0
                ELSE max(0.0, min(1.0,
                    1.0 - (
                        (raw_resource_cost - MIN(raw_resource_cost) OVER ())
                        / NULLIF(MAX(raw_resource_cost) OVER () - MIN(raw_resource_cost) OVER (), 0.0)
                    )
                ))
            END AS resource_cost_score
        FROM scored
    ),

    final_with_efficiency AS (
        SELECT
            *,
            CASE
                WHEN normalised_primary_metric IS NULL OR raw_resource_cost IS NULL THEN NULL
                ELSE normalised_primary_metric / (raw_resource_cost + 0.000000001)
            END AS cost_efficiency
        FROM final_values
    ),

    reviewed AS (
        SELECT
            *,
            CASE
                WHEN lower(COALESCE(status, '')) = 'failed' THEN 'failed'
                WHEN primary_metric IS NULL AND metric_score IS NULL THEN 'invalid_metric'
                WHEN primary_metric IS NULL THEN 'invalid_metric'
                WHEN metric_score IS NULL THEN 'invalid_metric'
                WHEN degenerate_zero_flag = 1 THEN 'degenerate_metric'
                WHEN required_auxiliary_missing_flag = 1 THEN 'degenerate_metric'
                WHEN (
                    (replace(COALESCE(primary_metric_name, ''), '-', '_') IN ('accuracy', 'f1', 'masked_accuracy') AND metric_score < 0.25)
                    OR (lower(COALESCE(task_type, '')) = 'text2text_generation' AND metric_score <= 0.10)
                    OR (replace(COALESCE(hf_task, ''), '-', '_') = 'causal_lm_generation' AND metric_score < 0.10)
                    OR (lower(COALESCE(task_type, '')) = 'detection' AND metric_score < 0.02)
                    OR (lower(COALESCE(task_type, '')) = 'segmentation' AND metric_score < 0.02)
                    OR (replace(COALESCE(hf_task, ''), '-', '_') = 'visual_question_answering' AND metric_score < 0.10)
                    OR (lower(COALESCE(task_type, '')) = 'image_captioning' AND metric_score < 0.10)
                ) THEN 'low_quality_review'
                ELSE 'ok'
            END AS review_bucket,
            CASE
                WHEN lower(COALESCE(status, '')) = 'failed' THEN 'service status is failed'
                WHEN primary_metric IS NULL AND metric_score IS NULL THEN 'missing primary metric and metric_score'
                WHEN primary_metric IS NULL THEN 'missing primary metric'
                WHEN metric_score IS NULL THEN 'missing metric_score'
                WHEN degenerate_zero_flag = 1 THEN 'degenerate zero-valued primary metric for ' || COALESCE(primary_metric_name, 'unknown')
                WHEN required_auxiliary_missing_flag = 1 THEN 'missing required secondary metric for primary metric ' || COALESCE(primary_metric_name, 'unknown')
                WHEN replace(COALESCE(primary_metric_name, ''), '-', '_') IN ('accuracy', 'f1', 'masked_accuracy') AND metric_score < 0.25
                    THEN 'metric_score below advisory threshold 0.25 for classification-style review'
                WHEN lower(COALESCE(task_type, '')) = 'text2text_generation' AND metric_score <= 0.10
                    THEN 'metric_score below advisory threshold 0.10 for text2text generation review'
                WHEN replace(COALESCE(hf_task, ''), '-', '_') = 'causal_lm_generation' AND metric_score < 0.10
                    THEN 'metric_score below advisory threshold 0.10 for text generation review'
                WHEN lower(COALESCE(task_type, '')) = 'detection' AND metric_score < 0.02
                    THEN 'metric_score below advisory threshold 0.02 for detection review'
                WHEN lower(COALESCE(task_type, '')) = 'segmentation' AND metric_score < 0.02
                    THEN 'metric_score below advisory threshold 0.02 for segmentation review'
                WHEN replace(COALESCE(hf_task, ''), '-', '_') = 'visual_question_answering' AND metric_score < 0.10
                    THEN 'metric_score below advisory threshold 0.10 for VQA review'
                WHEN lower(COALESCE(task_type, '')) = 'image_captioning' AND metric_score < 0.10
                    THEN 'metric_score below advisory threshold 0.10 for image captioning review'
                ELSE NULL
            END AS review_reason
        FROM final_with_efficiency
    )

    SELECT
        run_id,
        status,
        created_at,
        dataset,
        task_type,
        model_type,
        COALESCE(hf_model_id, 'Not Available') AS "HF model id",
        COALESCE(hf_dataset_id, 'Not Available') AS "HF dataset id",

        metric_score,
        COALESCE(primary_metric_name, 'Not Available') AS "Primary metric name",
        COALESCE(CAST(primary_metric AS TEXT), 'Not Available') AS "Primary metric",
        COALESCE(auxiliary_metric_name, 'Not Available') AS "Auxiliary metric name",
        COALESCE(CAST(auxiliary_metric AS TEXT), 'Not Available') AS "Auxiliary metric",
        COALESCE(CAST(latency AS TEXT), 'Not Available') AS "Latency",
        COALESCE(CAST(tail_latency AS TEXT), 'Not Available') AS "Tail latency",
        COALESCE(CAST(participation_rate AS TEXT), 'Not Available') AS "Participation rate",
        COALESCE(CAST(max(0.0, min(1.0, COALESCE(reliability_score, participation_rate))) AS TEXT), 'Not Available') AS "Reliability score",
        COALESCE(CAST(mean_compute_time AS TEXT), 'Not Available') AS "Mean compute time",
        COALESCE(CAST(resource_cost_score AS TEXT), 'Not Available') AS "Resource cost score",
        COALESCE(CAST(cost_efficiency AS TEXT), 'Not Available') AS "Cost efficiency",
        CASE
            WHEN model_size IS NULL OR model_size <= 0 THEN 'Not Available'
            ELSE CAST(model_size AS TEXT)
        END AS "Model size",
        COALESCE(CAST(downloads AS TEXT), 'Not Available') AS "Downloads",
        COALESCE(CAST(likes AS TEXT), 'Not Available') AS "Likes",
        COALESCE(CAST(hf_model_downloads AS TEXT), 'Not Available') AS "HF model downloads",
        COALESCE(CAST(hf_dataset_downloads AS TEXT), 'Not Available') AS "HF dataset downloads",
        COALESCE(CAST(hf_model_likes AS TEXT), 'Not Available') AS "HF model likes",
        COALESCE(CAST(hf_dataset_likes AS TEXT), 'Not Available') AS "HF dataset likes",
        COALESCE(CAST(learning_rate AS TEXT), 'Not Available') AS "Learning rate",
        COALESCE(CAST(batch_size AS TEXT), 'Not Available') AS "Batch size",
        COALESCE(split_strategy_requested, 'Not Available') AS "Split strategy requested",
        COALESCE(split_strategy_effective, 'Not Available') AS "Split strategy effective",
        COALESCE(split_skew_axis, 'Not Available') AS "Split skew axis",
        COALESCE(split_skew_axis_effective, 'Not Available') AS "Split skew axis effective",
        COALESCE(split_bucket_spec_json, 'Not Available') AS "Split bucket spec",
        COALESCE(data_distribution, 'Not Available') AS "Data distribution",
        COALESCE(dataset_distributions, 'Not Available') AS "Dataset distributions",
        COALESCE(CAST(dataset_size AS TEXT), 'Not Available') AS "Dataset size",
        COALESCE(CAST(explainability_score AS TEXT), 'Not Available') AS "Explainability score",
        COALESCE(update_signature_id, 'Not Available') AS "Update signature id",
        COALESCE(CAST(signature_dim AS TEXT), 'Not Available') AS "Signature dim",
        COALESCE(CAST(signature_norm AS TEXT), 'Not Available') AS "Signature norm",
        COALESCE(update_signature_path, 'Not Available') AS "Update signature path",
        COALESCE(update_signature_method, 'Not Available') AS "Update signature method",
        CASE
            WHEN lower(COALESCE(status, '')) = 'failed' THEN latest_failure_stage
            ELSE NULL
        END AS failure_stage,
        CASE
            WHEN lower(COALESCE(status, '')) = 'failed' THEN latest_failure_message
            ELSE NULL
        END AS failure_message,
        historical_failure_count,
        review_bucket,
        review_reason
    FROM reviewed
) flattened
WHERE review_bucket != 'ok'
ORDER BY
    CASE WHEN review_bucket = 'failed' THEN 0 ELSE 1 END,
    created_at DESC;
