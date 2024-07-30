import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle
import seaborn as sn
from sklearn.model_selection import StratifiedShuffleSplit, CalibratedClassifierCV
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from sklearn.base import BaseEstimator, ClassifierMixin



class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, batch_size=32, epochs=100):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        return self.model.predict(X)
    
class PredictionDistributionCallback(Callback):
    def __init__(self, validation_data, interval=1):
        super(PredictionDistributionCallback, self).__init__()
        self.validation_data = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.validation_data[0])
            print(f"\nEpoch {epoch}")
            print(f"Prediction distribution: Min={np.min(y_pred):.4f}, Max={np.max(y_pred):.4f}, Mean={np.mean(y_pred):.4f}")
            print(f"Predictions > 0.5: {np.sum(y_pred > 0.5)} / {len(y_pred)}") #Initially assumed threshold as 0.5


def data_init():
    event_embedd = pd.read_csv('*/event_embeddings_st.csv')
    member_embeddings = pd.read_csv('*/member_embeddings_node2vec_allg_1024.csv')
    combined_data = pd.read_csv('*/combined_data_membal.csv')

    return event_embedd, member_embeddings, combined_data


def load_files():
    with open('*/event_id_dict.pkl', 'rb') as f:
        event_id_dict = pickle.load(f)

    with open('*/event_id_dict_rev.pkl', 'rb') as f:
        event_id_dict_rev = pickle.load(f)

    with open('*/member_id_dict.pkl', 'rb') as f:
        member_id_dict = pickle.load(f)

    with open('*/member_id_dict_rev.pkl', 'rb') as f:
        member_id_dict_rev = pickle.load(f)
    
    return event_id_dict, event_id_dict_rev, member_id_dict, member_id_dict_rev

def data_preprocess(combined_data, event_embedd, member_embeddings):
    event_embedd.drop('Unnamed: 0', axis = 1, inplace = True)
    event_embed_np = event_embedd.to_numpy(dtype = 'float32')
    member_embeddings.drop(['Unnamed: 0'], axis = 1, inplace = True)
    member_embed_np = member_embeddings.to_numpy(dtype = 'float32')

    return combined_data, event_embedd, event_embed_np, member_embeddings, member_embed_np


def replace_data(combined_data, event_id_dict, event_embed_np, member_id_dict, member_embed_np):
    # Function to replace event_id with its corresponding numpy array
    def replace_event_id(event_id):
        idx = event_id_dict[event_id]
        return event_embed_np[idx]  # Default to zeros if not found

    # Function to replace member_id with its corresponding numpy array
    def replace_member_id(member_id):
        idx = member_id_dict[member_id]
        return member_embed_np[idx]  # Default to zeros if not found

    combined_data['event_id'] = combined_data['event_id'].apply(replace_event_id)
    combined_data['member_id'] = combined_data['member_id'].apply(replace_member_id)

    return combined_data

def data_generator(combined_data):
    event_list_embed = combined_data['event_id']
    member_list_embed = combined_data['member_id']
    label_list = combined_data['label']

    event_list_embed = np.array(combined_data['event_id'].tolist())
    member_list_embed = np.array(combined_data['member_id'].tolist())
    label_list = combined_data['label'].values 

    #Generating Data Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in sss.split(event_list_embed, label_list):
        X_train_event, X_test_event = event_list_embed[train_index], event_list_embed[test_index]
        X_train_member, X_test_member = member_list_embed[train_index], member_list_embed[test_index]
        y_train, y_test = label_list[train_index], label_list[test_index]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.111, random_state=42)
    for train_index, val_index in sss_val.split(X_train_event, y_train):
        X_train_event, X_val_event = X_train_event[train_index], X_train_event[val_index]
        X_train_member, X_val_member = X_train_member[train_index], X_train_member[val_index]
        y_train, y_val = y_train[train_index], y_train[val_index]

    return X_train_event, X_test_event, X_val_event, X_train_member, X_test_member, X_val_member, y_train, y_test, y_val

def reshape_if_needed(data):
    if len(data.shape) == 1:
        return data.reshape(-1, 1)
    return data

#Event Tower Architecture
def build_event_model(input_shape_event):
    input_event = Input(shape=input_shape_event)
    batch_norm_event = BatchNormalization()(input_event)
    dense_event_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(batch_norm_event)
    dropout_event_1 = Dropout(0.3)(dense_event_1)
    dense_event_2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(dropout_event_1)
    dropout_event_2 = Dropout(0.3)(dense_event_2)
    
    event_model = Model(inputs=input_event, outputs=dropout_event_2)
    return event_model

#Member Tower Architecture
def build_member_model(input_shape_member):
    input_member = Input(shape=input_shape_member)
    batch_norm_member = BatchNormalization()(input_member)
    dense_member_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(batch_norm_member)
    dropout_member_1 = Dropout(0.3)(dense_member_1)
    dense_member_2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(dropout_member_1)
    dropout_member_2 = Dropout(0.3)(dense_member_2)
    
    member_model = Model(inputs=input_member, outputs=dropout_member_2)
    return member_model

#Concatenating outputs of Member Tower and Event Tower
def build_combined_model(event_model, member_model):
    concatenated = concatenate([event_model.output, member_model.output])
    dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    dropout_1 = Dropout(0.3)(dense_1)
    dense_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dropout_1)
    dropout_2 = Dropout(0.3)(dense_2)
    output = Dense(1, activation='sigmoid')(dropout_2)
    
    combined_model = Model(inputs=[event_model.input, member_model.input], outputs=output)
    return combined_model

def model_compile(X_train_event, X_train_member):
    if isinstance(X_train_event, np.ndarray):
        input_shape_event = X_train_event.shape[1:]
    elif isinstance(X_train_event, list):
        input_shape_event = (len(X_train_event[0]),)
    else:
        raise ValueError(f"Unexpected type for X_train_event: {type(X_train_event)}")

    if isinstance(X_train_member, np.ndarray):
        input_shape_member = X_train_member.shape[1:]
    elif isinstance(X_train_member, list):
        input_shape_member = (len(X_train_member[0]),)
    else:
        raise ValueError(f"Unexpected type for X_train_member: {type(X_train_member)}")
    

    event_model = build_event_model(input_shape_event)
    member_model = build_member_model(input_shape_member)
    combined_model = build_combined_model(event_model, member_model)
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    combined_model.summary()
    return combined_model, event_model, member_model

#Training the model
def train(combined_model, event_model, member_model, X_train_event, X_val_event, X_train_member, X_val_member, y_train, y_val):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    pred_dist_callback = PredictionDistributionCallback(([X_val_event, X_val_member], y_val))
    
    class_weights = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    history = combined_model.fit(
        [X_train_event, X_train_member], y_train,
        validation_data=([X_val_event, X_val_member], y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, pred_dist_callback],
        verbose=1
    )
    
    combined_model.save('*/combined_model_final.h5')
    event_model.save('*/event_model_final.h5')
    member_model.save('*/member_model_final.h5')

    return history, combined_model, event_model, member_model

#Evaluation Metrics
def eval_crit(combined_model, X_test_event, X_test_member, y_test):
    # Adding Code for ROC Curve and Optimal Threshold Value
    y_pred_proba = combined_model.predict([X_test_event, X_test_member])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Finding the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal threshold:", optimal_threshold)

    # Evaluate the model and get the predictions
    loss, accuracy = combined_model.evaluate([X_test_event, X_test_member], y_test)
    y_pred_prob = combined_model.predict([X_test_event, X_test_member])
    y_pred = (y_pred_prob > optimal_threshold).astype(int)
    y_true = y_test

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print metrics
    print(f'Testing Loss: {loss}')
    print(f'Testing Accuracy: {accuracy}')
    print('Confusion Matrix:')
    cm = tf.math.confusion_matrix(labels = y_true, predictions = y_pred)
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot = True, fmt = 'd')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred))


#Plotting the training and loss value curve through the epochs
def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()



def main():
    #Initialise Relevant Data
    event_embedd, member_embeddings, combined_data = data_init()
    event_id_dict, event_id_dict_rev, member_id_dict, member_id_dict_rev = load_files()

    #Data Preprocessing
    combined_data, event_embedd, event_embed_np, member_embeddings, member_embed_np = data_preprocess(combined_data, event_embedd, member_embeddings)
    combined_data = replace_data(combined_data, event_id_dict, event_embed_np, member_id_dict, member_embed_np)

    #Splitting the Data into train, test, validation
    X_train_event, X_test_event, X_val_event, X_train_member, X_test_member, X_val_member, y_train, y_test, y_val = data_generator(combined_data)

    #Data Scaling
    scaler_event = StandardScaler()
    X_train_event = scaler_event.fit_transform(X_train_event)
    X_val_event = scaler_event.transform(X_val_event)
    X_test_event = scaler_event.transform(X_test_event)

    scaler_member = StandardScaler()
    X_train_member = scaler_member.fit_transform(X_train_member)
    X_val_member = scaler_member.transform(X_val_member)
    X_test_member = scaler_member.transform(X_test_member)

    X_train_event = reshape_if_needed(X_train_event)
    X_train_member = reshape_if_needed(X_train_member)
    X_val_event = reshape_if_needed(X_val_event)
    X_val_member = reshape_if_needed(X_val_member)
    X_test_event = reshape_if_needed(X_test_event)
    X_test_member = reshape_if_needed(X_test_member)

    #Compiling and Training the Model
    combined_model, event_model, member_model = model_compile(X_train_event, X_train_member)
    history, combined_model, event_model, member_model = train(combined_model, event_model, member_model, X_train_event, X_val_event, X_train_member, X_val_member, y_train, y_val)
    
    plot_training_history(history)
    
    # Load the saved models
    combined_model = load_model('*/combined_model_final.h5')
    event_model = load_model('*/event_model_final.h5')
    member_model = load_model('*/member_model_final.h5')
    
    combined_model.summary()
    event_model.summary()
    member_model.summary()

    eval_crit(combined_model, X_test_event, X_test_member, y_test)


    


if __name__== "__main__":
    main()
