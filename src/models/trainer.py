from typing import Any, Iterable, List

from matplotlib import pyplot as plt
import json
from sklearn.metrics import accuracy_score
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric


def calculate_charcter_accuracy(
    input: torch.LongTensor,
    target: torch.LongTensor,
    target_length: int
) -> float:
    """Рассчитать accuracy по декодированным предсказаниям"""
    decoded_input = []
    last_index = -1
    for char_ind in input:
        if char_ind != last_index and char_ind != 0:
            decoded_input.append(char_ind.item())
        last_index = char_ind
    target_length = max(target_length, len(decoded_input))
    target = target[:target_length]
    count_missing_chars = target_length - len(decoded_input)
    decoded_input.extend(count_missing_chars * [0])

    return (torch.LongTensor(decoded_input) == target.cpu()).numpy().mean()


class CRNNTrainer(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.criterion = torch.nn.CTCLoss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x = batch['image']
        target = batch['text']
        target_length = batch['text_length']
        input_length = batch['input_length'].view(-1)
        y = self(x)

        loss = self.criterion(y, target, input_length, target_length)
        accuracy = 0.
        y = torch.transpose(y, 0, 1)  # shape BS SEQ_LENGTH FEATURES
        y = y.argmax(-1)
        for pred_sample, target_sample, length in zip(y, target, target_length):
            accuracy += calculate_charcter_accuracy(pred_sample, target_sample, length)
        accuracy /= y.shape[1]

        return loss, accuracy

    def training_step(self, batch: Any, batch_idx: int):
        loss, accuracy = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/char_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, accuracy = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/char_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.test_conf_matrix(preds, targets.int())
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.log("test", self)
        # self.log("test")

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        confusion_matrix = self.test_conf_matrix.compute().cpu().numpy()
        self.test_conf_matrix.reset()
        for i, matrix in enumerate(confusion_matrix):
            fig, ax = plt.subplots()
            ax.imshow(matrix)
            self.logger.experiment.add_figure(f"confusion/class_{i}", fig, self.current_epoch)
            plt.close(fig)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if hasattr(self.hparams, 'lr'):
            optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CRNNTrainer(None, None, None)
