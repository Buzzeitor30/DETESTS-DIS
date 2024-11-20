import lightning as L
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class TransformerClassifier(L.LightningModule):
    """
    A PyTorch Lightning module for a transformer-based classifier.

    Args:
        transformer_classifier (nn.Module): The transformer-based classifier model.
        loss_fn (callable): The loss function used for training.
        lr (float): The learning rate for the optimizer.

    Attributes:
        transformer (nn.Module): The transformer-based classifier model.
        loss_fn (callable): The loss function used for training.
        lr (float): The learning rate for the optimizer.
    """

    def __init__(self, transformer_classifier, loss_fn, lr):
        super(TransformerClassifier, self).__init__()
        self.transformer = transformer_classifier
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, batch):
        return self.transformer(batch)

    def _common_step(self, batch):
        logits = self.transformer(batch)
        loss = self.loss_fn(logits, batch["label"]).mean()
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["label"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self._common_step(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["label"].shape[0],
        )
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.transformer.configure_optimizers(self.lr), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        logits = self.transformer(batch)
        return {"logits": logits, "id": batch["id"]}
