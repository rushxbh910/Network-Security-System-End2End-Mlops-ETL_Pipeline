import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import GPUtil
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object,
    load_object,
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("NetworkSecurity_Models")

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _log_metrics(self, stage: str, metrics_obj) -> None:
        """
        Generic MLflow metric logger: handles namedtuple or dict-like metrics.
        """
        if hasattr(metrics_obj, "_asdict"):
            items = metrics_obj._asdict().items()
        elif hasattr(metrics_obj, "metrics"):
            items = metrics_obj.metrics.items()
        else:
            items = vars(metrics_obj).items()

        for name, val in items:
            try:
                mlflow.log_metric(f"{stage}_{name}", float(val))
            except Exception:
                logging.warning(f"Could not log metric {stage}_{name}: {val}")

    def _log_gpu_metrics(self, stage: str) -> None:
        """
        Log GPU utilization and memory metrics to MLflow.
        """
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                mlflow.log_metric(f"{stage}_gpu_{i}_util", gpu.load)
                mlflow.log_metric(f"{stage}_gpu_{i}_memutil", gpu.memoryUtil)
                mlflow.log_metric(f"{stage}_gpu_{i}_memused", gpu.memoryUsed)
                mlflow.log_metric(f"{stage}_gpu_{i}_memfree", gpu.memoryFree)
        except Exception as e:
            logging.warning(f"Could not log GPU metrics at {stage}: {e}")

    def train_model(self, x_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
            "Random Forest": {'n_estimators': [8, 16, 32, 128, 256]},
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
        }

        # Evaluate and pick best model
        model_report = evaluate_models(
            X_train=x_train, y_train=y_train,
            X_test=x_test, y_test=y_test,
            models=models, params=params
        )
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # Infer signature for logging
        try:
            signature = infer_signature(x_train, best_model.predict(x_train))
            input_example = x_train[:5]
        except Exception:
            signature = None
            input_example = None

        # Start MLflow run
        with mlflow.start_run(run_name=best_model_name):
            # Log parameters
            mlflow.log_param("model_name", best_model_name)
            for p, v in best_model.get_params().items():
                mlflow.log_param(p, v)

            # Log GPU metrics pre-training
            self._log_gpu_metrics("pre_train")

            # Train on GPU if supported by the model
            best_model.fit(x_train, y_train)

            # Log GPU metrics post-training
            self._log_gpu_metrics("post_train")

            # Evaluate
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_metrics  = get_classification_score(y_true=y_test,  y_pred=y_test_pred)

            # Log metrics
            self._log_metrics("train", train_metrics)
            self._log_metrics("test",  test_metrics)

            # Log the pipeline model
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            pipeline = NetworkModel(preprocessor=preprocessor, model=best_model)

            mlflow.sklearn.log_model(
                sk_model=pipeline.model,
                artifact_path="sk_model",
                registered_model_name="NetworkSecurityModel",
                signature=signature,
                input_example=input_example
            )

        # Persist locally
        model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir, exist_ok=True)
        save_object(
            self.model_trainer_config.trained_model_file_path,
            obj=pipeline
        )
        logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metrics,
            test_metric_artifact=test_metrics
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)