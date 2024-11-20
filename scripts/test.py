import lightning as L
from lightning_models import TransformerClassifier
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from model import TransformerModel2
from dataset import TestDETESTSDataset
import pandas as pd 
import argparse
import os
import torch
from prediction_writer import CustomPredictionWriter
from run import get_model_args



def load_checkpoint(checkpoint_path, model_args):
    """Carga un modelo desde un checkpoint usando PyTorch Lightning."""
    return 


def get_num_neurons_annotators_depending_on_approach(approach):
    if approach == "hard":
        return 1, 1
    elif approach == "soft":
        return 1, 2
    elif approach == "annotators":
        return 3, 1

def create_dataloader(csv_path, batch_size=128):
    """Carga un archivo CSV y crea un DataLoader."""
    test_df = pd.read_csv(csv_path)
    test_dataset = TestDETESTSDataset(test_df)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    return dataloader

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


def main(run_name, task, approach):
    directory = os.path.join("logs", task, approach, run_name)
    test_dataloader = create_dataloader("test.csv")
    ckpt_files = filtrar_ficheros_ckpt(directory)
    #model_args = get_model_args(task, approach)

    for fold_id, ckpt in enumerate(ckpt_files, start=1):
        print("Predicting fold 1")



def predictions_from_hard_approach(preds):
    res_preds = {"id": [], "soft": [], "hard": []}

    res_preds["id"] = [id  for pred in preds for id in pred["id"]]
    aux = torch.sigmoid_(torch.cat([pred["logits"].squeeze(1) for pred in preds], dim=0))
    res_preds["soft"] = aux.tolist()
    res_preds["hard"] = (aux > 0.5).int().tolist()
    return pd.DataFrame(res_preds)

def predictions_from_soft_approach(preds):
    res_preds = {"id": [], "soft": [], "hard": []}
    res_preds["id"] = [id  for pred in preds for id in pred["id"]]

    aux = torch.softmax(torch.cat([pred["logits"] for pred in preds], dim=0), dim=1)
    res_preds["soft"] = aux.tolist()
    res_preds["hard"] = torch.argmax(aux, dim=1).tolist()
    return pd.DataFrame(res_preds)


if __name__ == "__main__":
    # Configuración del parser de argumentos
    parser = argparse.ArgumentParser(description="Cargar checkpoints de modelos usando PyTorch Lightning.")
    parser.add_argument("run_name", type=str, help="Nombre de la run")
    parser.add_argument("task", type=str, choices=["stereotype", "implicit"], help="Tipo de tarea")
    parser.add_argument("approach", type=str, choices=["hard", "soft", "annotators"], help="Enfoque")

    args = parser.parse_args()

    # Ejecutar el script principal con los argumentos proporcionados
    main(args.run_name, args.task, args.approach)
