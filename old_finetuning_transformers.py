import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
import argparse
import json
from pathlib import Path
import datasets
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datetime import datetime
from timeit import default_timer as timer
import subprocess

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
        "n_epochs": 0,
        "training_time": 0,
        "training_finished": False,
        "test_accuracy": 0
    }

    # name of model
    model_name = args.model_name
    print(f'\nusing model {model_name}\n')

    # create training dataset
    data_path = './data/' + args.dataset_name + '/'
    print(f'loading dataset from {data_path}')
    data_files = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    dataset = datasets.load_dataset(data_path, data_files=data_files)

    # load dataset info
    with open(data_path + 'info.json') as json_file:
        ds_info = json.load(json_file)
    
    print('dataset info:\n')
    for key, value in ds_info.items():
        if key != 'label_text_map':
            print(f'{key} : {value}')
    print('\n')

    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_dataset(data):
        return tokenizer(data['description'])

    dataset = dataset.map(tokenize_dataset)

    # Load the model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=ds_info['num_labels'])

    tf_train = model.prepare_tf_dataset(dataset['train'],batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)
    tf_val = model.prepare_tf_dataset(dataset['validation'],batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)
    tf_test = model.prepare_tf_dataset(dataset['test'],batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)

    print('\nresulting batches in each dataset:')
    print(f'train: {tf_train.cardinality().numpy()}\nvalidate: {tf_val.cardinality().numpy()}\ntest: {tf_test.cardinality().numpy()}\n')

    # configure callbacks
    
    # checkpoint
    filepath=f"./output/{args.dataset_name}/{model_name}/{date_str}_{time_str}/train/"
    # ensure directory exists
    Path(filepath).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(filepath + 'weights-best.hdf5', monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)
    
    # earlystopping
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.02, patience=4)
    
    # tensorboard
    tensorboard = TensorBoard(log_dir=f'./logs/{args.dataset_name}/{model_name}/{date_str}_{time_str}/')
    
    # custom callback for number of epochs
    # Define a custom callback to track the number of epochs trained
    class EpochCounterCallback(Callback):
        def on_train_begin(self, logs=None):
            self.epochs_trained = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epochs_trained += 1

    epoch_count = EpochCounterCallback()

    # custom callback for training time
    class TimingCallback(Callback):
        def __init__(self, logs={}):
            self.logs=[]
        def on_epoch_begin(self, epoch, logs={}):
            self.starttime = timer()
        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(timer()-self.starttime)

    training_timer = TimingCallback()


    callbacks_list = [checkpoint, early_stopping, tensorboard, epoch_count, training_timer]
    print(f'using callbacks {callbacks_list}\n')

    try:
        # train model
        # Lower learning rates are often better for fine-tuning transformers
        model.compile(optimizer=Adam(3e-5), metrics=['accuracy'])
        model.fit(tf_train, validation_data=tf_val, callbacks=callbacks_list, epochs=100)
        train_info["training_finished"] = True
        train_info["n_epochs"] = epoch_count.epochs_trained
        train_info["training_time"] += sum(training_timer.logs)
        print('finished training\n')
    except KeyboardInterrupt:
        train_info["training_finished"] = False
        train_info["n_epochs"] = epoch_count.epochs_trained
        train_info["training_time"] += sum(training_timer.logs)
    
    # evaluate test accuracy
    print('evaluating test accuracy:')
    score = model.evaluate(tf_test)
    train_info["test_accuracy"] = score[1]

    # saving training info
    out_path = f'./output/{args.dataset_name}/{model_name}/{date_str}_{time_str}/'
    with open(out_path + "training_info.json", "w+") as file:
        json.dump(train_info, file, indent=4)
    
    # save model
    
    print(f'\nsaving model at {out_path + "saved"}')
    # ensure directory exists
    Path(out_path).mkdir(parents=True, exist_ok=True)
    model.save(out_path)
