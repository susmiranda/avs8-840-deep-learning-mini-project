import torch
from torch import nn
import torch.nn.functional as F

from src.models.modules.EMA import EMA

'''
Code from: https://github.com/HolgerBovbjerg/data2vec-KWS/tree/main
'''

class SSTransformer(nn.Module):
    """
    Self Supervised Vision Transformer Module
    """
    def __init__(self,
                 encoder: nn.Module,
                 model_embed_dim: int,
                 ema_decay: float,
                 ema_end_decay: float,
                 ema_anneal_end_step: int,
                 average_top_k_layers: int,
                 normalize_targets: bool,
                 **kwargs):
        """
        :param encoder: transformer encoder
        :param modality: vision, audio or text
        :param model_embed_dim: Embedding dimension of transformer encoder
        :param ema_decay: EMA model decay
        :param ema_end_decay: EMA model end decay
        :param ema_anneal_end_step: Number of annealing steps for EMA model decay
        :param average_top_k_layers: Number of encoder layers to use for Data2Vec target
        :param normalize_targets: Specifies whether Dat2Vec targets are normalized
        :param kwargs: keyword arguments
        """
        super().__init__()
        self.encoder = encoder
        self.embed_dim = model_embed_dim
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.__dict__.update(kwargs)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.ema = EMA(self.encoder, device=device)  # Teacher
        self.regression_head = (
            self._build_regression_head()
        )  # Instantiate regression head to predict target

    def _build_regression_head(self):
        return nn.Linear(self.embed_dim, self.embed_dim)

    def ema_step(self):
        """
        Function which to step the EMA encoder
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, student_input, teacher_input=None, mask=None):
        """
        SS Transformer forward method.
        :param student_input: Input for student encoder
        :param teacher_input: Input for teacher encoder
        :param mask: mask for student input if input is not already masked
        :return: Data2Vec model output x, y for student prediction and teacher target, respectively
        """

        # The teacher input will usually be the same data as the student input.
        # Get the outputs of the model given the inputs
        encoder_out, student_hidden_states = self.encoder(student_input, mask=mask, output_hidden_states=True)        
        if teacher_input is None:
            return encoder_out
        
        # Adapted to timm

        # if teacher_input is None:
        #     encoder_out = self.encoder(student_input)
        #     return encoder_out
        # else:
        #     student_hidden_states = self.encoder.get_intermediate_layers(student_input)
        
        # If there is teacher input then we need to compare the latent space. Get the last hidden state from the transformer.
        x = student_hidden_states[-1]
        with torch.no_grad():
            self.ema.model.eval()

            # Also and without updating any gradients get the latent space of the EMA model.
            # teacher_hidden_states = self.ema.model.get_intermediate_layers(teacher_input, self.average_top_k_layers)
            _, teacher_hidden_states = self.ema.model(teacher_input, mask=None, output_hidden_states=True)

            # Average out the last k hidden states of the EMA model? This is now our target. 
            y = teacher_hidden_states[-self.average_top_k_layers:]
            
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)
            if self.normalize_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

        x = x[mask]
        y = y[mask]

        x = self.regression_head(x)

        return x, y