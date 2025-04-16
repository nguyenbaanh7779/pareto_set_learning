import itertools
import os

import abc
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass

from enum import Enum
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Set, Union, cast

import numpy as np
import torch
from torch import Tensor
from datasets import Dataset, load_dataset
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert import BertForSequenceClassification
from transformers import AutoTokenizer, BertTokenizer, PreTrainedTokenizer

from _common import RESULTS_DIR, logging
from fusionlib.merge.task_arithmetic import task_arithmetic_merge_modules
from src.module.dict_moe import ParetoWeightEnsemblingModule
from src.module.utils import print_trainable_parameters
from src.phn.solvers import EPOSolver
from src.utils import timeit_context

# Configure log.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)


@dataclass
class InputExample:
    uid: str


@dataclass
class TextClassificationExample(InputExample):
    doc_tokens: List[str]
    label: Optional[str] = None
    positions: Optional[List[List[int]]] = None


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    positions: Optional[List[List[int]]] = None


@dataclass
class TextClassificationFeatures(InputFeatures):
    label: Optional[int] = None


class Mode(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    inference = "inference"


def load_imdb_dataset(split: Mode) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Load IMDB dataset.
    :param split: Train or Test split.
    :return: Dataset in dictionary format and list of all labels of the dataset.
    """
    log.info(f"Load IMDB dataset from {split} split")
    # TODO: We convert the dataset into format {"text": [], "labels": []} which was used commonly. This step can require
    #  a large MEM as the dataset is duplicated.
    imdb_dataset = load_dataset("imdb")[split.value]
    output_dataset: Dict[str, List[str]] = {"text": [], "labels": []}
    all_labels: Set[str] = set()
    for sample in tqdm(imdb_dataset):
        output_dataset["text"].append(sample["text"])
        label = str(sample["label"])
        output_dataset["labels"].append(label)
        all_labels.add(label)
    log.info(f"Loaded dataset with {len(all_labels)} labels")
    return output_dataset, list(all_labels)


# TODO: This method is a copy from the one in `scripts.gpt2_pareto_moe` module. We should move it into a common place.
#  However, this module will be shared with other member so it is okay to keep it here.
def generate_simplex_grid(n, m):
    """
    Generate a uniform grid of points on the n-dimensional simplex.

    Args:
        n (int): The dimension of the simplex.
        m (int): The number of grid points along each dimension.

    Returns:
        list: A list of n-dimensional vectors representing the grid points.
    """
    m = m - 1
    # **Generate all combinations of indices summing up to m**
    indices = list(itertools.combinations_with_replacement(range(m + 1), n - 1))

    # **Initialize an empty list to store the grid points**
    grid_points = []

    # **Iterate over each combination of indices**
    for idx in indices:
        # **Append 0 and m to the indices**
        extended_idx = [0] + list(idx) + [m]

        # **Compute the vector components by taking the differences between consecutive indices and dividing by m**
        point = [(extended_idx[i + 1] - extended_idx[i]) / m for i in range(n)]
        grid_points.append(point)

    return np.array(grid_points, dtype=np.float32)


class Processor(ABC):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, str],
        max_seq_len: int,
        label_list: List[str],
        **kwargs,
    ):
        if isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, **kwargs, use_fast=False
            )
        self.max_seq_len = max_seq_len
        self.label_list = label_list

    @abc.abstractclassmethod
    def convert_examples_to_features(
        self, examples: List[InputExample]
    ) -> List[InputFeatures]:
        """Generate input features from examples"""
        raise NotImplementedError()

    @abc.abstractclassmethod
    def features_to_dataset(
        self, features: List[InputFeatures], mode: Union[str, Mode]
    ) -> Dataset:
        """Get Pytorch Dataset object from list of input features"""
        raise NotImplementedError()


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


class TextClassificationProcessor(Processor):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, str],
        max_seq_len: int,
        label_list: List[str],
        multilabel: bool = False,
        quotechar: str = '"',
        skiprows: int = 1,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            **kwargs,
        )
        self.quotechar = quotechar
        self.skiprows = skiprows
        self.multilabel = multilabel

    def get_examples(self, training_data: Dict) -> List[TextClassificationExample]:
        """
        convert training data to bert examples
        :param training_data: dict{"text": text, "labels", labels}
        """
        assert len(training_data["text"]) == len(training_data["labels"]), (
            f"{len(training_data['text'])} text and {len(training_data['labels'])} labels"
        )
        examples = []
        is_contain_position = "positions" in training_data
        for ii in tqdm(range(len(training_data["text"]))):
            label = training_data["labels"][ii]
            assert label in self.label_list, (
                f"Non exist label: '{label}' in label list: {self.label_list}."
            )
            context_text = training_data["text"][ii]
            # List of tokens of the doc.
            doc_tokens: List[str] = []
            # Mapping between position of the character to the word position.
            char_to_word_offset = []
            prev_is_whitespace = True
            for cc in context_text:
                if is_whitespace(cc):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(cc)
                    else:
                        doc_tokens[-1] += cc
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            if is_contain_position:
                assert len(training_data["positions"][ii]) == len(doc_tokens)
                positions = training_data["positions"][ii]
            else:
                positions = [[0, 0, 0, 0]] * len(doc_tokens)
            examples.append(
                TextClassificationExample(
                    uid=f"{ii}", doc_tokens=doc_tokens, label=label, positions=positions
                )
            )
        return examples

    def _convert_example_to_feature(
        self, example: TextClassificationExample, label_to_idx: Dict[str, int]
    ) -> TextClassificationFeatures:
        all_positions = []
        all_doc_tokens = []
        for ii, token in enumerate(example.doc_tokens):
            sub_tokens = self.tokenizer.tokenize(token)
            all_doc_tokens.extend(sub_tokens)
            all_positions.extend([example.positions[ii]] * len(sub_tokens))
        encoded_dict = self.tokenizer.encode_plus(
            all_doc_tokens,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_token_type_ids=True,
        )
        position_pad = [0, 0, 0, 0]
        if len(all_positions) <= self.max_seq_len - 2:
            positions = [position_pad] + all_positions
        else:
            positions = [position_pad] + all_positions[: self.max_seq_len - 2]

        if len(positions) < self.max_seq_len:
            positions += [position_pad] * (self.max_seq_len - len(positions))
        encoded_dict["positions"] = positions
        encoded_dict["label"] = label_to_idx[example.label]
        return TextClassificationFeatures(**encoded_dict)

    def convert_examples_to_features(
        self, examples: List[TextClassificationExample]
    ) -> List[TextClassificationFeatures]:
        """Generate text classification features from examples"""
        label_to_idx = {label: ii for ii, label in enumerate(self.label_list)}
        features: List[TextClassificationFeatures] = []
        for ii in tqdm(range(len(examples))):
            features.append(
                self._convert_example_to_feature(examples[ii], label_to_idx)
            )
        return features

    def features_to_dataset(
        self,
        features: List[TextClassificationFeatures],
        mode: Union[str, Mode] = Mode.train,
    ) -> TensorDataset:
        """Get Pytorch Dataset object from list of classification features"""
        if isinstance(mode, Mode):
            mode = mode.value
        dataset = [
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            torch.tensor([f.positions for f in features], dtype=torch.long),
            torch.tensor([f.label for f in features], dtype=torch.long),
        ]

        return TensorDataset(*dataset)


class BERTParetoMoEProgram(ABC):
    def __init__(self, cfg: DictConfig, __file__: str):
        self.finetuned_backbone = None
        self.pretrained_model: Optional[BertForSequenceClassification] = None
        self.finetuned_models: Dict[str, BertForSequenceClassification] = {}
        self.cfg = cfg
        if cfg.model is None:
            raise ValueError("model must be specified")

        self.result_dir = (
            RESULTS_DIR
            / os.path.basename(__file__).split(".")[0]
            / f"version_{cfg.version}"
        )

        # Setup fabric.
        self.fabric = Fabric(
            accelerator="cuda",
            loggers=TensorBoardLogger(
                save_dir=self.result_dir, name="tb_logs", version=""
            ),
            strategy=DDPStrategy() if cfg.num_devices > 1 else "auto",
            callbacks=[DeviceStatsMonitor(), LearningRateMonitor("step")],
        )
        self.fabric.launch()
        # Attributes which will be initialized later.
        self.epo_solver = None
        self.train_loader_iters = None
        self.task_num_labels_mapping = None
        self.train_loaders = None

    def run(self):
        cfg = self.cfg

        self.load_datasets()
        self.load_model()

        if cfg.train:
            self.train()

    def load_datasets(self):
        """
        Load datasets
        """

        cfg = self.cfg

        assert cfg.batch_size % cfg.num_devices == 0, (
            "Batch size must be divisible by num_devices"
        )

        cfg.batch_size = cfg.batch_size // cfg.num_devices

        log.info("Loading datasets")

        tokenizer = BertTokenizer.from_pretrained(self.cfg.model)

        # TODO: For testing purpose, we keep both task similar. We need to change this in actual experiment.
        # TODO: This should be configurable.
        dataset_mapping = {
            "task1": load_imdb_dataset(Mode.train),
            "task2": load_imdb_dataset(Mode.train),
        }

        def get_dataloader(data: Dict[str, Any], labels: List[str]) -> DataLoader:
            dataprocessor = TextClassificationProcessor(
                tokenizer=tokenizer, max_seq_len=512, label_list=labels
            )
            log.info("Load examples")
            examples = dataprocessor.get_examples(data)
            log.info("Convert examples to features")
            features = dataprocessor.convert_examples_to_features(examples)
            log.info("Construct dataset")
            dataset = dataprocessor.features_to_dataset(features)
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=False if cfg.num_devices > 1 else True,
                sampler=(
                    DistributedSampler(dataset, shuffle=True)
                    if cfg.num_devices > 1
                    else None
                ),
            )
            return dataloader

        self.train_loaders = {
            task_name: get_dataloader(*data)
            for task_name, data in dataset_mapping.items()
        }
        # Mapping between task name and the number of labels in that task.
        # TODO: Generally the number of labels should be different between tasks, but in this very simple experiment
        #  we set them the same value.
        self.task_num_labels_mapping = {
            task_name: len(data[1]) for task_name, data in dataset_mapping.items()
        }
        self.train_loader_iters = [
            iter(itertools.cycle(dd)) for dd in self.train_loaders.values()
        ]

    def load_model(self):
        """
        Load pretrained model and finetuned models.
        """
        cfg = self.cfg

        log.info(
            "Loading pretrained model. The classification layer can be different between tasks but in this simple "
            "experiment, we use the same value."
        )
        self.pretrained_model = BertForSequenceClassification.from_pretrained(
            self.cfg.model, num_labels=list(self.task_num_labels_mapping.values())[0]
        )

        pretrained_backbone = self.pretrained_model.bert

        for task in cfg.tasks:
            log.info(f"Loading finetuned model for task {task}")
            self.finetuned_models[task] = BertForSequenceClassification.from_pretrained(
                self.cfg.model, num_labels=self.task_num_labels_mapping[task]
            )

        # Store the finetuned backbone.
        self.finetuned_backbone: Dict[str, nn.Module] = {
            task: cast(BertLayer, model.bert)
            for task, model in self.finetuned_models.items()
        }

        # Clean the backbone of the finetuned model.
        for task in cfg.tasks:
            self.finetuned_models[task].bert = None

        with timeit_context("building model"):
            if self.cfg.partial:
                # Weight ensembling only the MLPs, merge the remaining layers using task arithmetic
                # model merging.
                model: nn.Module = task_arithmetic_merge_modules(
                    pretrained_backbone,
                    list(self.finetuned_backbone.values()),
                    scaling_coef=cfg.init_lambda,
                )
                # Do not update the model weights.
                model.requires_grad_(False)

                # We apply weight merging on dense layer of each BertLayer.
                for layer_idx in range(self.pretrained_model.config.num_hidden_layers):
                    cast(
                        BertLayer, model.encoder.layer[layer_idx]
                    ).output.dense = ParetoWeightEnsemblingModule(
                        base_model=model.encoder.layer[layer_idx].output.dense,
                        expert_models=[
                            self.finetuned_backbone[task]
                            .encoder.layer[layer_idx]
                            .output.dense
                            for task in self.finetuned_backbone.keys()
                        ],
                        init_lambda=cfg.init_lambda,
                        fix_base_model_and_experts=True,
                        router_hidden_layers=cfg.router_hidden_layers,
                    )
            else:
                raise NotImplementedError(
                    "Full model weight ensembling not implemented"
                )

        self.model = model

        print_trainable_parameters(self.model)

    def compute_loss(self, model: nn.Module, ray: Tensor, losses: List[Tensor]):
        """
        Compute loss.
        """
        # TODO: This is a "runnable" version, we are not sure the purpose and meaning of this function.
        if self.epo_solver is None:
            num_objectives = len(self.finetuned_models)
            self.epo_solver = EPOSolver(n_tasks=num_objectives, n_params=None)
        epo_solver = self.epo_solver

        losses = torch.stack(losses)
        loss = epo_solver.get_weighted_loss(
            losses, ray, tuple(filter(lambda p: p.requires_grad, model.parameters()))
        )
        return loss

    def train(self):
        """
        A "runnable" training function.
        """
        cfg = self.cfg
        device = self.fabric.device
        log.info("Training")

        # Save the configuration.
        self.result_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.result_dir / "train_config.yaml")

        # Set up the model.
        num_objectives = len(self.finetuned_backbone)
        backbone = deepcopy(self.model)

        print(self.model)
        classifiers = {
            task: cast(BertForSequenceClassification, m)
            .classifier.requires_grad_(False)
            .to(device)
            for task, m in self.finetuned_models.items()
        }
        log.info("Classifiers:")
        for task, classifier in classifiers.items():
            log.info(f"{task}: {classifier}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, backbone.parameters()), lr=cfg.lr
        )

        backbone, optimizer = self.fabric.setup(backbone, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.num_steps, eta_min=cfg.lr * 0.1
        )

        # Setup forward models which share a common backbone.
        forward_model = deepcopy(self.pretrained_model)
        forward_model.requires_grad_(False)
        forward_model.bert = backbone
        forward_model.to(device)

        backbone.train()
        for step_idx in tqdm(
            range(1, 1 + cfg.num_steps), "training", dynamic_ncols=True
        ):
            # Sample a preference ray.
            ray = torch.from_numpy(
                np.random.dirichlet((cfg.alpha,) * num_objectives, 1)
                .astype(np.float32)
                .flatten()
            ).to(device)
            ParetoWeightEnsemblingModule.set_preferenec_vector(backbone, ray)

            losses: List[Tensor] = []
            for dataset_idx, dataset_name in enumerate(cfg.tasks):
                batch = next(self.train_loader_iters[dataset_idx])
                forward_model.num_labels = self.finetuned_models[
                    dataset_name
                ].num_labels

                outputs = torch.func.functional_call(
                    forward_model,
                    parameter_and_buffer_dicts={
                        "classifier." + k: v
                        for k, v in classifiers[dataset_name]
                        .state_dict(keep_vars=True)
                        .items()
                    },
                    args=tuple(),
                    kwargs=dict(
                        input_ids=batch[0].to(device),
                        attention_mask=batch[1].to(device),
                        labels=batch[-1].to(device),
                    ),
                    strict=False,
                )

                losses.append(outputs.loss)

            loss = self.compute_loss(backbone, ray, losses)

            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            lr_scheduler.step()

           # Log overall loss
            self.fabric.log("loss", loss.item(), step=step_idx)

            # Log per-task losses
            for i, task_name in enumerate(cfg.tasks):
                self.fabric.log(f"{task_name}_loss", losses[i].item(), step=step_idx)

            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            self.fabric.log("learning_rate", current_lr, step=step_idx)

            # Log gradient norm
            total_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in backbone.parameters() if p.grad is not None
            ]), 2)
            self.fabric.log("grad_norm", total_grad_norm.item(), step=step_idx)

            # # Log L2 distance to each expert
            # def l2_distance(model1, model2):
            #     return sum(torch.norm(p1 - p2).item()
            #             for p1, p2 in zip(model1.parameters(), model2.parameters()))

            # for task, expert in self.finetuned_backbone.items():
            #     dist = l2_distance(backbone, expert)
            #     self.fabric.log(f"distance_to_{task}", dist, step=step_idx)

            if step_idx % cfg.save_interval == 0:
                (self.result_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
                self.fabric.save(
                    self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt",
                    {"model": backbone},
                )


if __name__ == "__main__":
    config = DictConfig(
        {
            "model": "bert-base-uncased",
            "version": None,
            "num_devices": 1,
            "tasks": {"task1": 1, "task2": 1},
            "partial": True,
            "init_lambda": 0.6,
            "router_hidden_layers": 1,
            "batch_size": 1,
            "train": True,
            "lr": 1e-2,
            "num_steps": 1000,
            "alpha": 1,
            "save_interval": 500,
        }
    )
    program = BERTParetoMoEProgram(config, __file__)
    program.run()
