import pytest

from mlaas_data_generator.models.builders import create_model


def test_create_model_clustering_defaults_none_k_to_three():
    model = create_model(
        input_shape=(4,),
        num_classes=1,
        task_type="clustering",
        model_type="kmeans",
        clustering_k=None,
    )

    assert model.k == 3


def test_create_model_clustering_rejects_non_positive_k():
    with pytest.raises(ValueError, match="clustering_k must be a positive integer"):
        create_model(
            input_shape=(4,),
            num_classes=1,
            task_type="clustering",
            model_type="kmeans",
            clustering_k=0,
        )
