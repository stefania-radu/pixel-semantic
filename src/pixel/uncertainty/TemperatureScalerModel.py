import os
import torch
from torch import nn, optim
from torch.nn import functional as F


# source: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

class TemperatureScalerModel(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, trainer):
        super(TemperatureScalerModel, self).__init__()
        self.trainer = trainer
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.nll_before = None
        self.ece_before = None
        self.nll_after= None
        self.ece_after = None

    def get_metrics(self):
        return {
            'nll_before': self.nll_before,
            'ece_before': self.ece_before,
            'nll_after': self.nll_after,
            'ece_after': self.ece_after
        }
    
    def get_temperature(self):
        return self.temperature

    def forward(self, input):
        outputs = self.trainer.predict(input)
        # logits = outputs.predictions
        logits = torch.tensor(outputs.predictions).cuda()
        logits = logits.view(-1, 9)
        predictions = self.temperature_scale(logits).view(-1, 256, 9).cpu().detach().numpy()
        label_ids = outputs.label_ids
        
        return predictions, label_ids, outputs.metrics

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        # print(self.temperature)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


    # This function probably should live outside of this class, but whatever
    def set_temperature(self, calibration_dataset):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        """

        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        with torch.no_grad():
            outputs = self.trainer.predict(calibration_dataset, metric_key_prefix="calib")
            logits = torch.tensor(outputs.predictions).cuda()  # Convert numpy array to tensor
            labels = torch.tensor(outputs.label_ids).cuda()

        logits = logits.view(-1, 9)
        labels = labels.view(-1)

        print(f"logits.shape {logits.shape }") # torch.Size([5848, 256, 9])
        print(f"labels.shape {labels.shape}") # torch.Size([5848, 256])

        # Calculate NLL and ECE before temperature scaling
        self.nll_before = nll_criterion(logits, labels).item()
        self.ece_before = ece_criterion(logits, labels).item()
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.02, max_iter=1000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        self.nll_after = nll_criterion(self.temperature_scale(logits), labels).item()
        self.ece_after = ece_criterion(self.temperature_scale(logits), labels).item()

        
        
        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        # model_filename = os.path.join("src/pixel/uncertainty/models", 'model_with_temperature.pth')
        # torch.save(self.model.state_dict(), model_filename)
        # self.trainer.save_model(model_filename)
        # print('Temperature scaled model saved to %s' % model_filename)
        # print('Done!')

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece