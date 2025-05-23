import torch.nn as nn

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class AbstractModel(nn.Module):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(AbstractModel, self).__init__()

        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def calculate_loss(self, batch):
        raise NotImplementedError('calculate_loss method must be implemented.')

    def generate(self, batch, n_return_sequences=1):
        raise NotImplementedError('predict method must be implemented.')
