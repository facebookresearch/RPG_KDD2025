import torch


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        return results
