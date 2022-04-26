from torch.utils.data import DataLoader
import torch

import ipdb
import numpy as np
import wandb

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 500 # 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        ############ 2.8 TODO
        # Should return your validation accuracy
        num_batches = len(self._val_dataset_loader)
        losses = []
        ii = 0
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            ii += 1
            if ii > 500:
                break
            self._model.eval() 
            predicted_answer = self._model(batch_data["image"].to(device = ("cuda" if torch.cuda.is_available() else "cpu")),
                                                batch_data["question"].to(device = ("cuda" if torch.cuda.is_available() else "cpu")))
            # todo: majority vote for answer
            ground_truth_answer = batch_data["answers"].to(device = ("cuda" if torch.cuda.is_available() else "cpu"))
            # predicted_answer_imax = torch.argmax(predicted_answer)
            # loss is really accuracy
            # print(predicted_answer.shape, ground_truth_answer.shape)
            # print(torch.argmax(predicted_answer, dim=1) , torch.argmax(ground_truth_answer, dim=1))
            loss = (torch.argmax(predicted_answer, dim=1) == torch.argmax(ground_truth_answer, dim=1))
            # print(loss)
            loss = loss.float().mean().cpu()
            # loss = torch.nn.CrossEntropyLoss()(predicted_answer, ground_truth_answer).item()
            # print(loss)
            losses.append(loss)
            del batch_data
            torch.cuda.empty_cache()
            # printimg = batch_data["image"][3]
            # printquestion = batch_data["question"][3]
            # printanswer = batch_data["answer"][3]
            

        ############

        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            pass

            ############
        return np.mean(losses)
        # raise NotImplementedError()

    def train(self):

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.

                predicted_answer = self._model(batch_data["image"].to(device = ("cuda" if torch.cuda.is_available() else "cpu")),
                                                 batch_data["question"].to(device = ("cuda" if torch.cuda.is_available() else "cpu")))
                # todo: majority vote for answer
                ground_truth_answer = batch_data["answers"].to(device = ("cuda" if torch.cuda.is_available() else "cpu"))

                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)
                del batch_data
                torch.cuda.empty_cache()

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    wandb.log({"train_loss": loss})


                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    wandb.log({"val_acc": val_accuracy})

                    ############
