from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.disc_optimizer.zero_grad()
            self.gen_optimizer.zero_grad()

        gen_results = self.generator(**batch)
        batch.update(gen_results)

        fake_mel_updates = {'fake_mel': self.make_mel(batch['fake'].squeeze(1))}
        batch.update(fake_mel_updates)

        mpd_d_pass = self.mpd(**batch, detach=True)
        batch.update(mpd_d_pass)
        msd_d_pass = self.msd(**batch, detach=True)
        batch.update(msd_d_pass)

        disc_losses = self.disc_loss(**batch)
        batch.update(disc_losses)

        if self.is_train:
            batch["disc_loss"].backward()
            self._clip_grad_norm()
            self.disc_optimizer.step()

        mpd_g_pass = self.mpd(**batch, detach=False)
        batch.update(mpd_g_pass)
        msd_g_pass = self.msd(**batch, detach=False)
        batch.update(msd_g_pass)

        gen_losses = self.gen_loss(**batch)
        batch.update(gen_losses)
        
        if self.is_train:
            batch["gen_loss"].backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        
        self.writer.add_audio("fake_audio", batch['fake'][0], 22050)
        self.writer.add_audio("real_audio", batch['real'][0], 22050)
        self.writer.add_image("fake_mel", batch['fake_mel'][0].T)
        self.writer.add_image("real_mel", batch['real_mel'][0].T)

