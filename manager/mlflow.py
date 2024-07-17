from mlflow.tracking import MlflowClient


class MLFlowManager:
    def __init__(self, name: str = "default"):
        self.client = MlflowClient()
        self.name = name

    def __enter__(self):
        if self.client.get_experiment_by_name(self.name):
            experiment = self.client.get_experiment_by_name(self.name)
            if experiment:
                self.id = experiment.experiment_id
        else:
            self.id = self.client.create_experiment(self.name)
        self.run_id = self.client.create_run(self.id).info.run_id
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.set_terminated(self.run_id)

    def log_param(self, key, params):
        self.client.log_param(self.run_id, key, params)

    def log_metric(self, key, metrics, epoch):
        self.client.log_metric(self.run_id, key, metrics, step=epoch)
