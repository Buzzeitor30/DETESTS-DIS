import argparse
import pandas as pd
from torch.utils.data import DataLoader
import os
from model import TransformerModel
from dataset import DETESTSDataset, TestDETESTSDataset
from loss_functions import (
    BCEWithLogitsLossMultitask,
    CEWithLogitsLoss,
    BCEWithLogitsLoss,
    KLDivLoss,
)
from utils import get_kf_splits, preprocess_text, set_torch_np_random_rseed
from torch.utils.data import Subset
import lightning as L
from lightning_models import TransformerClassifier
from prediction_writer import CustomPredictionWriter
import evaluation


def set_up_argparse():
    parser = argparse.ArgumentParser(
        prog="run.py", description="Script to launch experimets"
    )

    # Add argument for run_name
    parser.add_argument(
        "run_name",
        help="Name of this run",
    )
    # Add argument for task
    parser.add_argument(
        "task",
        choices=["stereotype", "implicit"],
        default="stereotype",
        help="Specify the task (stereotype or implicit)",
    )
    # Add argument for approach
    parser.add_argument(
        "--approach",
        choices=["hard", "soft", "annotators"],
        default="hard",
        help="Specify the approach (hard, soft, or annotators)",
    )
    # Add argument for learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Set up learning rate",
    )
    # Add argument for weight decay
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-2,
        help="Set up weight decay",
    )
    # Huggingface model
    parser.add_argument(
        "--hfm",
        default="PlanTL-GOB-ES/roberta-base-bne",
        help="Hugging Face model name",
    )

    # Add argument for optimizer
    parser.add_argument(
        "--opt",
        choices=["adamw", "adam"],
        default="adamw",
        help="Specify the optimizer (adamw or adam)",
    )

    # Add boolean flag for context
    parser.add_argument(
        "--context",
        action="store_true",
        help="Add context to the data",
    )
    # Epsilon decay if LoRA
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=1,
        help="Epsilon decay for each layer in BERT",
    )
    parser.add_argument(
        "--mlp_hidden_neurons",
        help="Hidden neurons in classifier head",
        default=128,
        type=int,
    )

    # Epoch number
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train each fold"
    )
    # Loss fnct
    parser.add_argument(
        "--criterion",
        default="",
        help="Select criterion function",
        choices=["bce", "ce", "kl", "multibce", ""],
    )
    # Output neurons MLP
    parser.add_argument(
        "--out_neurons", default=1, help="Select output neurons of MLP", type=int
    )
    # Cross validatioon size
    parser.add_argument(
        "--cross_val",
        default=5,
        type=int,
        help="How many folds in cross Validation experiment",
    )
    # Batch size
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for the DataLoader"
    )
    # Preprocess data
    parser.add_argument(
        "--preprocess", action="store_true", help="Apply preprocessing to the data"
    )
    # Augmented data
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Apply data augmentation by back translation (ES-ENG-ES)",
    )
    parser.add_argument(
        "--max-length",
        default=128,
        type=int,
        help="Max length of the input text",
    )
    # device
    parser.add_argument(
        "--device", type=int, default=0, help="Specify device for Torch GPU"
    )
    parser.add_argument(
    "--test", action="store_true", help="Test the model")
    return parser


def get_model_args(argos):
    model_name = argos.hfm
    decay = argos.epsilon_decay
    mlp_output_size = argos.out_neurons
    num_annotators = 1
    hidden_layer_mlp = argos.mlp_hidden_neurons

    model_init_params = {
        "huggingface_model_name": model_name,
        "decay": decay,
        "hidden_layer": hidden_layer_mlp,
        "max_length": argos.max_length,
    }

    if argos.approach == "annotators":
        num_annotators = 3

    model_init_params["output_neurons_mlp"] = mlp_output_size
    model_init_params["num_annotators"] = num_annotators

    return model_init_params


def create_folder(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print("Error! Model folder already exists")


def get_criterion(argos):

    criterion_name = argos.criterion
    criterion_map = {
        "ce": CEWithLogitsLoss(),
        "bce": BCEWithLogitsLoss(),
        "kl": KLDivLoss(),
        "multi": BCEWithLogitsLossMultitask(),
    }
    criterion = criterion_map.get(criterion_name, None)
    if criterion is None:
        if argos.approach == "hard":
            criterion = BCEWithLogitsLoss()
        elif argos.approach == "soft":
            criterion = CEWithLogitsLoss()
        else:
            criterion = BCEWithLogitsLossMultitask()

    return criterion


def create_csv_logger(save_dir, run_name, version):
    return L.pytorch.loggers.CSVLogger(save_dir, name=run_name, version=version)


def create_tensorboard_logger(save_dir, run_name, version):
    return L.pytorch.loggers.TensorBoardLogger(save_dir, name=run_name, version=version)


def create_loggers(save_dir, run_name, version):
    return [
        create_csv_logger(save_dir, run_name, version),
        create_tensorboard_logger(save_dir, run_name, version),
    ]


def create_callbacks(dirpath):
    return [
        L.pytorch.callbacks.EarlyStopping(
            "val_loss", stopping_threshold=1e-3, patience=3
        ),
        L.pytorch.callbacks.ModelCheckpoint(dirpath, monitor="val_loss"),
    ]


def create_prediction_writer(output_dir, write_interval, filename, approach, task):
    return CustomPredictionWriter(output_dir, write_interval, filename, approach, task)


def create_dataframe(csv_path="train.csv", preprocess=False):
    df = pd.read_csv(
        csv_path
    )  # .groupby("stereotype", group_keys=False).apply(lambda x: x.sample(frac=.15))
    # Preprocessing
    if preprocess:
        df["text"] = df.apply(lambda sample: preprocess_text(sample), axis=1)
    # Stereotype
    df["stereotype_hard"] = df["stereotype"].astype(float)
    df["stereotype_annotators"] = (
        df[["stereotype_a1", "stereotype_a2", "stereotype_a3"]]
        .values.astype(float)
        .tolist()
    )
    #df["stereotype_soft"] = [[1 - x, x] for x in df["stereotype_soft"]]
    # Implicit
    df["implicit_hard"] = df["implicit"].astype(float)
    df["implicit_annotators"] = (
        df[["implicit_a1", "implicit_a2", "implicit_a3"]].values.astype(float).tolist()
    )
    #df["implicit_soft"] = [[1 - x, x] for x in df["implicit_soft"]]
    return df


def train(argos):
    # Extract args
    run_name = args.run_name
    task = args.task
    approach = args.approach
    # Opt settings
    learning_rate = args.lr
    weight_decay = args.wd
    optimizer_name = args.opt
    # Model settings
    model_name = args.hfm
    # Epochs to train
    epochs = args.epochs
    # Dataset
    context = args.context
    # Batch size
    batch_size = args.batch_size
    # Cross val size
    cross_val_size = args.cross_val
    # Device
    device = args.device

    path = "train.csv"
    if args.aug:
        path = "train_augmented.csv"
    df = create_dataframe(path, preprocess=args.preprocess)

    splitter = get_kf_splits(df, task + "_" + approach, n_splits=cross_val_size)

    directory = os.path.join("logs", task, approach)
    L.seed_everything(42)
    for fold, (train_idx, test_idx) in enumerate(splitter, start=1):

        print(f"Fold {fold}")
        # Model creation
        model_args = get_model_args(args)
        model = TransformerModel(**model_args)
        # Criterion
        criterion = get_criterion(args)
        # Create Datasets
        all_dataset = DETESTSDataset(df, add_context=args.context, task=task, approach=approach)
        # Data loader
        train_loader = DataLoader(
            Subset(all_dataset, train_idx),
            batch_size,
            shuffle=True,
            num_workers=8,
        )
        dev_loader = DataLoader(
            Subset(all_dataset, test_idx),
            batch_size,
            shuffle=False,
            num_workers=8,
        )
        loggers = create_loggers(directory, run_name, f"Fold_{fold}")
        callbacks = create_callbacks(os.path.join(directory, run_name, f"Fold_{fold}"))
        callbacks.append(
            create_prediction_writer(
                os.path.join(directory, run_name, f"Fold_{fold}"),
                "epoch",
                f"{run_name}_fold{fold}",
                approach,
                task,
            )
        )
        # Trainer and L_model
        trainer = L.Trainer(
            devices=[device],
            max_epochs=epochs,
            accumulate_grad_batches=32 // batch_size,
            logger=loggers,
            callbacks=callbacks,
            enable_model_summary=False if fold > 1 else True,
        )
        L_model = TransformerClassifier(model, criterion, learning_rate)
        print("FIT!")
        trainer.fit(L_model, train_loader, dev_loader)

        print("PREDICT!")
        trainer.predict(L_model, dev_loader, ckpt_path="best")
    print("GET METRICS!")
    directory = os.path.join(directory, run_name)

    preds_files = sorted(
        [
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if f.endswith(".json")
        ]
    )
    gold_file = path.split(".")[0] + "_" + task

    hard_df, soft_df = evaluation.evaluate_task(
        preds_files, gold_file, task, os.path.join(directory, "current_gold.json")
    )

    hard_df.to_csv(os.path.join(directory, f"results_{run_name}_hard.csv"), index=False)
    soft_df.to_csv(os.path.join(directory, f"results_{run_name}_soft.csv"), index=False)

def filtrar_ficheros_ckpt(directorio):
    # Lista para almacenar los archivos .ckpt encontrados
    ficheros_ckpt = []
    # Recorrer todos los directorios y subdirectorios
    for raiz, _, archivos in os.walk(directorio):
        # Filtrar los archivos que terminan en .ckpt
        for fichero in archivos:
            if fichero.endswith('.ckpt'):
                # Agregar el camino completo del archivo a la lista
                ficheros_ckpt.append(os.path.join(raiz, fichero))

    return sorted(ficheros_ckpt)

def test(argos):
    # Extract args
    run_name = argos.run_name
    task = argos.task
    approach = argos.approach
    #Default settings for model
    criterion = get_criterion(args)
    lr = None
    #Directory, extract CPKT per fold and model args
    directory = os.path.join("/home/elural/DETESTS-DIS/scripts/logs/", task, approach, run_name)
    ckpt_files = filtrar_ficheros_ckpt(directory)
    model_args = get_model_args(argos)

    #dataset
    df = pd.read_csv("test_solutions.csv")
    dataset = TestDETESTSDataset(df)
    test_Dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=10)

    for fold_id, ckpt_path in enumerate(ckpt_files, start=1):
        print(f"Predicting fold {fold_id}")
        model = TransformerModel2(**model_args)
        L_model = TransformerClassifier.load_from_checkpoint(ckpt_path, map_location=f"cuda:{argos.device}", transformer_classifier=model, loss_fn=criterion, lr=lr)
        callbacks = [
                create_prediction_writer(
                os.path.join(directory, f"Fold_{fold_id}"),
                "epoch",
                f"{run_name}_fold{fold_id}_test",
                approach,
                task,
            )
        ]
        trainer = L.Trainer(
            devices=[argos.device],
            callbacks=callbacks,
        )
        trainer.predict(L_model, test_Dataloader)

    preds_files = sorted(
        [
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if "_test" in f and f.endswith(".json")
        ]
    )

    gold_file = "test" + "_" + task

    hard_df, soft_df = evaluation.evaluate_task(
        preds_files, gold_file, task, os.path.join(directory, "current_gold.json")
    )

    hard_df.to_csv(os.path.join(directory, f"results_{run_name}_hard_test.csv"), index=False)
    soft_df.to_csv(os.path.join(directory, f"results_{run_name}_soft_test.csv"), index=False)

if __name__ == "__main__":
    parser = set_up_argparse()
    args = parser.parse_args()
    if args.test is True:
        test(args)
    else:
        train(args)
