import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TextClassificationPipeline
from pathlib import Path
from timeit import default_timer as timer


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

class Model(object):
    def __init__(self, name, dataset, batch_size, learning_rate, loss, max_epochs, train_info):
        self.name = name
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_epochs = max_epochs
        self.train_info = train_info
        self.model = None
        self.transformer_based = False

        # variables established from dataset object
        self.n_classes = dataset.info['num_labels']

    def generate_vectorizer(self):
        vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=self.dataset.vocab_size)
        vectorizer.adapt(self.dataset.full.map(lambda description, labels: description))

        return vectorizer

    def build_glove_matrix(self, glove_file, embedding_dim):
        path_to_glove_file = f'data/glove/{glove_file}'

        embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        print(f"\nFound {len(embeddings_index)} word vectors.\n")
        # print(f'embedding examples:\n')
        # print(list(embeddings_index.items())[0:3])

        # Prepare embedding matrix
        voc = self.vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        print(f'shape of embedding matrix: {embedding_matrix.shape}')
        hits = 0
        miss = 0
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                miss += 1
        print(f"Converted {hits} words ({miss} misses)")

        return embedding_matrix, word_index

    def set_callbacks(self, out_path_model, log_path, es_min_delta=0.02, es_patience=5):
        # checkpoint
        # ensure directory exists
        Path(out_path_model).mkdir(parents=True, exist_ok=True)
        if not self.transformer_based:
            self.checkpoint = ModelCheckpoint(out_path_model + '{epoch:02d}-{val_loss:.2f}', monitor='val_accuracy', mode='max', save_weights_only=True)
        else:
            # transformer models need to be saved as .hdf5
            self.checkpoint = ModelCheckpoint(out_path_model + '{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_accuracy', mode='max', save_weights_only=True)
        
        # earlystopping
        self.early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=es_min_delta, patience=es_patience)
        
        # tensorboard
        self.tensorboard = TensorBoard(log_dir=log_path)
        
        # custom callback for number of epochs
        # Define a custom callback to track the number of epochs trained
        class EpochCounterCallback(Callback):
            def on_train_begin(self, logs=None):
                self.epochs_trained = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epochs_trained += 1

        self.epoch_count = EpochCounterCallback()

        # custom callback for training time
        class TimingCallback(Callback):
            def __init__(self, logs={}):
                self.logs=[]
            def on_epoch_begin(self, epoch, logs={}):
                self.starttime = timer()
            def on_epoch_end(self, epoch, logs={}):
                self.logs.append(timer()-self.starttime)

        self.training_timer = TimingCallback()


        self.callbacks_list = [self.checkpoint, self.early_stopping, self.tensorboard, self.epoch_count, self.training_timer]
        print(f'using callbacks {self.callbacks_list}\n')

    def train(self, is_experiment=False):
        if not self.transformer_based:
            # shuffle and batch the dataset
            self.tf_train = self.dataset.train.shuffle(buffer_size=10000).batch(self.batch_size)
            self.tf_val = self.dataset.val.shuffle(buffer_size=10000).batch(self.batch_size)
            
            # compile model
            self.model.compile(loss=self.loss,
                optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                metrics=['accuracy'])
        else:
            # compile model without loss for transformer based models
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                metrics=['accuracy'])
        
        try:
            # train and save history
            self.history = self.model.fit(self.tf_train, validation_data=self.tf_val, callbacks=self.callbacks_list, epochs=self.max_epochs)
            if not is_experiment:
                self.train_info["training_finished"] = True
                self.train_info["n_epochs"] = self.epoch_count.epochs_trained
                self.train_info["training_time"] = sum(self.training_timer.logs)
        except KeyboardInterrupt:
            if not is_experiment:
                self.train_info["training_finished"] = False
                self.train_info["n_epochs"] = self.epoch_count.epochs_trained
                self.train_info["training_time"] = sum(self.training_timer.logs)
           
    def evaluate(self, is_experiment=False):
        if not self.transformer_based:
            # prepare test data
            self.tf_test = self.dataset.test.shuffle(buffer_size=10000).batch(self.batch_size)
        # evaluate test accuracy
        print('evaluating test accuracy:')
        score = self.model.evaluate(self.tf_test)

        # calculate sklearn classification report
        # make predictions using the model
        if self.transformer_based:
            # predicted_probs = self.predict(self.dataset.test_features)
            report = None
        else:
            predicted_probs = self.model.predict(self.dataset.test_features)
            # convert probabilities to class labels
            predicted_labels = tf.argmax(predicted_probs, axis=1).numpy()
            # convert true labels from the dataset to a numpy array
            true_labels = self.dataset.test_labels
            # generate the classification report
            report = classification_report(true_labels, predicted_labels)

        if not is_experiment:
            self.train_info["test_accuracy"] = score[1]
            self.train_info["classification_report"] = report
        else:
            return score[1], report

    def run_experiment(self, n_runs=5):
        # define training info
        self.train_info["experiment"] = []
        print(f'\n\nruning experiment for {n_runs} runs')

        for i in range(1, n_runs+1):
            print(f"\n--> experiment run numer {i}\n")
            # resead to get different initialisation in each run
            tf.keras.utils.set_random_seed(42+i)

            # set callbacks with apropriate paths
            out_path_model = f"./output/{self.dataset.name}/{self.name}/{self.train_info['date_str']}_{self.train_info['time_str']}_experiment/train_{i}/"
            log_path = f"./logs/{self.dataset.name}/{self.name}/{self.train_info['date_str']}_{self.train_info['time_str']}_experiment/run_{i}/"
            self.set_callbacks(out_path_model, log_path)

            # run train with experiment set to true
            self.train(is_experiment=True)
            test_acc, report = self.evaluate(is_experiment=True)
            
            train_dict = {
                "training_finished": True,
                "n_epochs": self.epoch_count.epochs_trained,
                "training_time": sum(self.training_timer.logs),
                "test_accuracy": test_acc,
                "classificatoin_report": report
            }
            self.train_info["experiment"].append(train_dict)
        
        # finally save training_info to json
        out_path_training_info = f"./output/{self.dataset.name}/{self.name}/{self.train_info['date_str']}_{self.train_info['time_str']}_experiment/"
        self.save_training_info(out_path_training_info)

    def show_training_graphs(self):
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plot_graphs(self.history, 'accuracy')
        plt.subplot(1, 2, 2)
        plot_graphs(self.history, 'loss')

    def save_training_info(self, out_path):
        with open(out_path + "training_info.json", "w+") as file:
            json.dump(self.train_info, file, indent=4)




class BiLSTM(Model):
    def __init__(self, dataset, batch_size, learning_rate, loss, max_epochs, train_info):
        name = 'BiLSTM'
        super().__init__(name, dataset, batch_size, learning_rate, loss, max_epochs, train_info)

        # run class functions to instanciate encoder and model
        self.vectorizer = self.generate_vectorizer()
        self.embedding_matrix, self.word_index = self.build_glove_matrix('glove.6B/glove.6B.50d.txt', embedding_dim=50)
        self.model = self.instanciate_model()

    def instanciate_model(self, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50):
        model = tf.keras.Sequential([
                self.vectorizer,
                tf.keras.layers.Embedding(len(self.word_index) + 1,
                                          EMBEDDING_DIM,
                                          weights=[self.embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=True),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.2)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(self.n_classes, activation='softmax')
            ])
        
        return model
        
class RCNN(Model):
    def __init__(self, dataset, batch_size, learning_rate, loss, max_epochs, train_info):
        name = 'RCNN'
        super().__init__(name, dataset, batch_size, learning_rate, loss, max_epochs, train_info)

        # run class functions to instanciate encoder and model
        self.vectorizer = self.generate_vectorizer()
        self.embedding_matrix, self.word_index = self.build_glove_matrix('glove.6B/glove.6B.50d.txt', embedding_dim=50)
        self.model = self.instanciate_model()
    
    def instanciate_model(self,MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50):

        kernel_size = 2
        filters = 64
        pool_size = 2
        gru_node = 32


        model = tf.keras.Sequential([
            self.vectorizer,
            tf.keras.layers.Embedding(len(self.word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(gru_node, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(gru_node, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(gru_node, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(gru_node, dropout=0.2)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
            ])

        return model
    
class Transformer(Model):
    def __init__(self, name, dataset, batch_size, learning_rate, loss, max_epochs, train_info):
        super().__init__(name, dataset, batch_size, learning_rate, loss, max_epochs, train_info)
        self.transformer_based = True

        # load tokenizer
        self.load_tokenizer()
        # tokenize dataset
        self.tokenize_dataset()
        # load model
        self.load_model()
        # prepare dataset
        self.prepare_dataset()    
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    def tokenize_dataset(self):
        def _tokenize_dataset(data):
            return self.tokenizer(data['description'])
        
        self.dataset.dataset = self.dataset.dataset.map(_tokenize_dataset)

    def prepare_dataset(self):
        self.tf_train = self.model.prepare_tf_dataset(self.dataset.dataset['train'],batch_size=self.batch_size, shuffle=True, tokenizer=self.tokenizer)
        self.tf_val = self.model.prepare_tf_dataset(self.dataset.dataset['validation'],batch_size=self.batch_size, shuffle=True, tokenizer=self.tokenizer)
        self.tf_test = self.model.prepare_tf_dataset(self.dataset.dataset['test'],batch_size=self.batch_size, shuffle=True, tokenizer=self.tokenizer)

    def load_model(self):
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.name, num_labels=self.n_classes)

    def predict(self, features, top_k=1, return_scores=False):
        # build pipeline for predictions
        pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=top_k)
        # predict labels for features
        predictions = pipe(features)
        if return_scores:
            return predictions
        else:
            # extract only labels from pipeline results
            pred_labels = [x['label'] for x in predictions]
            return pred_labels
