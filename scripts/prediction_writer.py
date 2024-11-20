from lightning.pytorch.callbacks import BasePredictionWriter
import torch
import numpy as np
import pandas as pd


class CustomPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, filename, approach, task):
        super(CustomPredictionWriter, self).__init__(write_interval)
        self.output_dir = output_dir
        self.write_interval = write_interval
        self.filename = filename
        self.approach = approach
        self.task = task
        self.task_keys = (
            ["NoStereotype", "Stereotype"]
            if task == "stereotype"
            else ["Explicit", "Implicit"]
        )

        self.soft_df = {"id": [], "value": []}
        self.hard_df = {"id": [], "value": []}

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):

        logits = outputs["logits"]
        id = outputs["id"]
        if self.approach == "hard":
            preds = torch.sigmoid(logits).squeeze().detach().cpu()
            preds = torch.stack((1 - preds, preds), dim=1).numpy()
        elif self.approach == "soft":
            preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
        else:
            preds = torch.where(
                torch.sigmoid(logits).detach().cpu() > 0.5, 1.0, 0.0
            )  # swap logits by 0 or 1
            preds = torch.sum(preds, dim=1, keepdim=True)  # sum all votes
            preds = torch.hstack((3 - preds, preds))  # create a tensor with the votes
            preds = torch.softmax(preds, dim=1).numpy()  # Softmax to get probabilities

        hard_preds = np.argmax(preds, axis=1)
        preds = [dict(zip(self.task_keys, p)) for p in preds.tolist()]

        self.soft_df["value"].extend(preds)
        self.soft_df["id"].extend(list(map(str, id)))

        self.hard_df["value"].extend([self.task_keys[p] for p in hard_preds])
        self.hard_df["id"].extend(list(map(str, id)))

    def on_predict_epoch_end(self, trainer, pl_module):
        self.soft_df = pd.DataFrame(self.soft_df)
        self.hard_df = pd.DataFrame(self.hard_df)
        self.hard_df["test_case"] = "DETESTS-Dis"
        self.soft_df["test_case"] = "DETESTS-Dis"

        self.soft_df.to_json(
            f"{self.output_dir}/{self.filename}_soft.json", orient="records"
        )
        self.hard_df.to_json(
            f"{self.output_dir}/{self.filename}_hard.json", orient="records"
        )
