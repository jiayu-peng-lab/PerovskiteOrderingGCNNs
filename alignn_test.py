from inference.select_best_models import reverify_wandb_models, keep_the_best_few_models
from inference.test_model_prediction import get_all_model_predictions
from inference.embedding_extraction import get_all_embeddings

get_all_model_predictions(
    model_params={
        "struct_type": "relaxed",
        "model_type": "ALIGNN",
        "training_fraction":1.0,
    },
    gpu_num=0,
    num_best_models=3
)