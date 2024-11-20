import torch
from torch import nn
from peft import LoraConfig, LoraModel
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class AnnotatorHead(nn.Module):
    """This class represents the classification head to append at the end of our Transformer"""

    def __init__(self, in_neurons, hidden_neurons, out_neurons):
        """Init function for FC layer of annotator

        Args:
            in_neurons int: Input neurons
            out_neurons int: Output neurons in last layer
        """
        super(AnnotatorHead, self).__init__()
        self.fc = torch.nn.Sequential(
            nn.Linear(in_features=in_neurons, out_features=hidden_neurons),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=hidden_neurons, out_features=out_neurons),
        )

    def forward(self, x):
        return self.fc(x)
    
class TransformerModel(nn.Module):
    """
    A PyTorch module for a Transformer-based model.

    Args:
        huggingface_model_name (str): The name of the Hugging Face model to use.
        decay (float): The decay factor for the learning rate.
        num_annotators (int): The number of annotators.
        **kwargs: Additional keyword arguments.

    Attributes:
        encoder (AutoModel): The Hugging Face model encoder.
        hidden_size (int): The size of the hidden layer.
        num_annotators (int): The number of annotators.
        hidden_layer (int): The size of the hidden layer in the annotator heads.
        annotators (nn.ModuleList): A list of annotator heads.

    Methods:
        forward(batch): Performs a forward pass through the model.
        configure_optimizers(learning_rate): Configures the optimizers for training.

    """

    def __init__(self, huggingface_model_name, decay, num_annotators, **kwargs):
        super(TransformerModel2, self).__init__()
        self.encoder = AutoModel.from_pretrained(huggingface_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_annotators = num_annotators
        self.max_length = kwargs.get("max_length", 512)
        self.hidden_layer = kwargs.get("hidden_layer", 128)
        self.output_neurons = kwargs.get("output_neurons_mlp", 1)
        self.annotators = nn.ModuleList(
            [AnnotatorHead(self.hidden_size, self.hidden_layer, self.output_neurons) for _ in range(num_annotators)]
        )
        self.decay = decay

    def forward(self, batch):
        text = self.tokenizer(batch["text"][0], text_pair=batch["text"][1], padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to("cuda:1")
        x = self.encoder(**text).last_hidden_state[:, 0]
        y = [ann(x) for ann in self.annotators]
        return torch.cat(y, dim=1)
    
    def configure_optimizers(self, learning_rate):
        config_parameters_dynamic = [{"params": self.annotators.parameters()}]
        last_lr = learning_rate
        for layer in self.encoder.encoder.layer[::-1]:
            config_parameters_dynamic.append(
                {"params": layer.parameters(), "lr": last_lr}
            )
            last_lr = self.decay * last_lr
        return config_parameters_dynamic