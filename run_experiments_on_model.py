import json
import pandas as pd
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
from model_classes import BiLSTM, RCNN, Transformer
from dataset_classes import Dataset, TransformerDataset


parser = argparse.ArgumentParser(
    description="script for running dataset experiments on models that are not transformer based.")

parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-m',
                        dest="model_name",
                        help="name/model id of transformer model. available models hosted inside a model repo on huggingface.co.",
                        type=str,
                        required=True)
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-ba',
                        dest="batch_size",
                        help="int defining the batch size for training",
                        default=16,
                        type=int)
optional.add_argument('-lr',
                        dest="learning_rate",
                        help="float defining the learning rate used for training, for transformer models 3e-5 is recommended",
                        default=0.001,
                        type=float)

def run_experiment(model_name, dataset, train_info, n_runs, results_df):
    train_info["experiment"] = []

    # generate list to fill with results and append to dataframe at the end
    results_acc = []
    results_time = []

    print(f'\n\nruning experiment for {n_runs} runs')

    for i in range(1, n_runs+1):
        print(f"\n--> experiment run numer {i}\n")
        
        # reseed tensorflow to get new initialisation
        tf.keras.utils.set_random_seed(42+i)

        # instanciate model
        if model_name == 'BiLSTM':
            model = BiLSTM(dataset=dataset, batch_size=args.batch_size, learning_rate=args.learning_rate, loss='categorical_crossentropy', max_epochs=100, train_info=train_info)
        elif model_name == 'RCNN':
            model = RCNN(dataset=dataset, batch_size=args.batch_size, learning_rate=args.learning_rate, loss='categorical_crossentropy', max_epochs=100, train_info=train_info)
        else:
            print(f'\nassuming <{model_name}> is transformer based. Name must be a valid huggingface model identifier listed on "https://huggingface.co/models".\n')
            try:
                model = Transformer(name=model_name, dataset=dataset, batch_size=args.batch_size, learning_rate=args.learning_rate, loss=None, max_epochs=100, train_info=train_info)
            except OSError:
                print(f'\ninvalid model name <{model_name}> specified. Must be one of [BiLSTM, RCNN] or a valid huggingface model identifier listed on "https://huggingface.co/models".\n')
                exit(1)

        # training process
        out_path_model = f"./output/{dataset.name}/{model_name}/{train_info['date_str']}_{train_info['time_str']}_experiment/train_{i}/"
        log_path = f"./logs/{dataset.name}/{model_name}/{train_info['date_str']}_{train_info['time_str']}_experiment/run_{i}/"
        model.set_callbacks(out_path_model, log_path)
        model.train(is_experiment=True)
        test_acc, report = model.evaluate(is_experiment=True)
        
        # add to results lists
        results_acc.append(test_acc)
        results_time.append(sum(model.training_timer.logs))
        # build training dict for this run
        train_dict = {
            "training_finished": True,
            "n_epochs": model.epoch_count.epochs_trained,
            "training_time": sum(model.training_timer.logs),
            "test_accuracy": test_acc,
            "classificatoin_report": report
        }
        train_info["experiment"].append(train_dict)
        
    # finally safe training info
    out_path_training_info = f"./output/{dataset.name}/{model_name}/{train_info['date_str']}_{train_info['time_str']}_experiment/"
    with open(out_path_training_info + "training_info.json", "w+") as file:
            json.dump(train_info, file, indent=4)

    # add results to dataframe
    results_df[f'{dataset.name}_acc'] = results_acc
    results_df[f'{dataset.name}_time'] = results_time
    

if __name__ == "__main__":
    # get arguments
    args = parser.parse_args()

    # save current date and time for directory output
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    time_str = today.strftime("%H%M%S")

    # define datasets for experiment
    dataset_names = ['00_baseline', '01_corrupted', '02_duplicates', '03_mislabels', '04_class_imbalance']

    # name of model
    model_name = args.model_name
    print(f'\nusing model {model_name}\n')

    # generate dataframe to be filled with results
    results_df = pd.DataFrame()

    for dataset_name in dataset_names:
        print(f'running experiment for {dataset_name}\n')
        # define dict with training info
        train_info = {
            "dataset": dataset_name,
            "model": args.model_name,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "used_gpu": str(subprocess.check_output(["nvidia-smi", "-L"])),
            "start_date_time": today.strftime("%Y%m%d_%H%M%S"),
            "date_str": date_str,
            "time_str": time_str,
        }

        # read in dataset
        if model_name not in ['BiLSTM', 'RCNN']:
            ds = TransformerDataset(dataset_name)
        else:
            ds = Dataset(dataset_name)

        run_experiment(model_name, dataset=ds, train_info=train_info, n_runs=5, results_df=results_df)
    # safe results dataframe
    results_df.to_csv(f'output/experiment_results/{model_name}_{date_str}_{time_str}.csv', index=False)

    print (f'finished experiment for all datasets on {model_name}!')