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
    description="script for running finetuning with dataset on BERT model")

parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-d',
                        dest="dataset_name",
                        help="directory containing the prepared dataset csv files",
                        type=str,
                        required=True)
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

if __name__ == "__main__":
    # get arguments
    args = parser.parse_args()

    # save current date and time for directory output
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    time_str = today.strftime("%H%M%S")

    # define dict with training info
    train_info = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "batch_size": args.batch_size,
        "used_gpu": str(subprocess.check_output(["nvidia-smi", "-L"])),
        "start_date_time": today.strftime("%Y%m%d_%H%M%S"),
        "date_str": date_str,
        "time_str": time_str,
        "training_finished": False,
    }

    # name of model
    model_name = args.model_name
    print(f'\nusing model {model_name}\n')

    # read in dataset
    if model_name not in ['BiLSTM', 'RCNN']:
        ds = TransformerDataset(args.dataset_name)
    else:
        ds = Dataset(args.dataset_name)

    # instanciate model
    if model_name == 'BiLSTM':
        model = BiLSTM(dataset=ds, batch_size=args.batch_size, learning_rate=args.learning_rate, loss='categorical_crossentropy', max_epochs=100, train_info=train_info)
    elif model_name == 'RCNN':
        model = RCNN(dataset=ds, batch_size=args.batch_size, learning_rate=args.learning_rate, loss='categorical_crossentropy', max_epochs=100, train_info=train_info)
    else:
        print(f'\nassuming <{model_name}> is transformer based. Name must be a valid huggingface model identifier listed on "https://huggingface.co/models".\n')
        try:
            model = Transformer(name=model_name, dataset=ds, batch_size=args.batch_size, learning_rate=args.learning_rate, loss=None, max_epochs=100, train_info=train_info)
        except OSError:
            print(f'\ninvalid model name <{model_name}> specified. Must be one of [BiLSTM, RCNN] or a valid huggingface model identifier listed on "https://huggingface.co/models".\n')
            exit(1)

    # training process
    out_path_model = f"./output/{ds.name}/{model_name}/{date_str}_{time_str}/train/"
    log_path = f"./logs/{ds.name}/{model_name}/{date_str}_{time_str}/"
    model.set_callbacks(out_path_model, log_path)
    model.train()
    model.evaluate()
    out_path_info = f"./output/{ds.name}/{model_name}/{date_str}_{time_str}/"
    model.save_training_info(out_path_info)

    # run experiment
    #model.run_experiment(n_runs=3)
    