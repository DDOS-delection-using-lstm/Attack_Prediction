# train_model.py
import numpy as np
 
import joblib
import os,json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,ReduceLROnPlateau # type: ignore
from sklearn.utils.class_weight import compute_class_weight
import logging
from dataloader import load_cicddos2019

# Configure logging
logging.basicConfig(
    filename='train_model.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
 
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         logger.error(f"GPU setup error: {str(e)}")
from tensorflow.keras.utils import plot_model

def save_model_architecture(model):
    """Save a plot of the model architecture"""
    os.makedirs('model', exist_ok=True)
    plot_model(model, to_file='model/model_architecture.png', show_shapes=True, show_layer_names=True, dpi=150)

def prepare_data(data_df=None, data_path=None):
   
    if data_df is None and data_path:
        data_df = load_cicddos2019(data_path)
    
    if data_df is None:
        raise ValueError("No valid dataset provided.")
 
    label_col = None
    for col in data_df.columns:
        if any(keyword in col.lower() for keyword in [' label', 'attack', 'class']):
            label_col = col
            break

    if not label_col:
        raise ValueError("Could not identify the label column in the dataset.")
 
    drop_cols = [' Timestamp', ' Fwd Header Length']   
    drop_cols = [col for col in drop_cols if col in data_df.columns]
    data_df.drop(columns=drop_cols, errors='ignore', inplace=True)

    label_encoder = LabelEncoder()
    data_df[label_col] = label_encoder.fit_transform(data_df[label_col])

    unique_labels = np.unique(data_df[label_col])
    if set(unique_labels) != {0, 1}:
        raise ValueError(f"Label column must contain only 0 or 1, but found: {unique_labels}")

    for col in ['Source IP', ' Destination IP']:
        if col in data_df.columns:
            data_df[col] = data_df[col].astype(str).apply(lambda x: int(hash(x) % 100000))
             
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_df.dropna(inplace=True)
    
    for col in data_df.columns:
        if data_df[col].dtype == 'O':   
            print(f"Dropping non-numeric column: {col}")
            data_df.drop(columns=[col], inplace=True)
 
    numeric_cols = data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != label_col] 
    
    feature_cols = [col for col in data_df.columns if col != label_col]
    scaler = StandardScaler()
    data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])

    scaler = StandardScaler()
    data_df[numeric_cols] = scaler.fit_transform(data_df[numeric_cols])

    return data_df, scaler, label_encoder, feature_cols, label_col



def create_balanced_sequences(data, label_col, seq_length=10, max_samples_per_class=10000):
     
    features = data.drop(label_col, axis=1).astype(np.float32)
    target = data[label_col].to_numpy()
    features_np = features.to_numpy(copy=False)
    
    valid_range = features_np.shape[0] - seq_length
    class_0_idx = np.where(target[:valid_range] == 0)[0]
    class_1_idx = np.where(target[:valid_range] == 1)[0]
    
    samples_per_class = min(len(class_0_idx), len(class_1_idx), max_samples_per_class // 2)
    if samples_per_class <= 0:
        raise ValueError("Not enough samples in one or both classes to create sequences.")
 
    class_0_idx = np.random.choice(class_0_idx, samples_per_class, replace=False)
    class_1_idx = np.random.choice(class_1_idx, samples_per_class, replace=False)
  
    X, y = [], []
    for idx in class_0_idx:
        seq = features_np[idx : idx + seq_length]
        X.append(seq)
        y.append(0)
    for idx in class_1_idx:
        seq = features_np[idx : idx + seq_length]
        X.append(seq)
        y.append(1) 
    X = np.array(X)
    y = np.array(y)
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    return X, y


def build_model(input_shape):
    
    
    reg_factor = 0.01
    regularizer = tf.keras.regularizers.l2(reg_factor)
    
    model = Sequential([
         LSTM(32,
         input_shape=input_shape,
         return_sequences=True,
         kernel_regularizer=regularizer,
         recurrent_regularizer=regularizer,
         dropout=0.5,
         recurrent_dropout=0.3),
        BatchNormalization(),
    
        LSTM(16,
         return_sequences=False,
         kernel_regularizer=regularizer,
         recurrent_regularizer=regularizer,
         dropout=0.5,
         recurrent_dropout=0.3),
         BatchNormalization(),
    
        Dense(8, activation='relu', kernel_regularizer=regularizer),
            BatchNormalization(),
            Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

    return model

def evaluate_model(model, X, y, dataset_name='Test'):
    """Evaluate model and save confusion matrix"""
    y_pred = (model.predict(X) > 0.5).astype(int)
    
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'model/confusion_matrix_{dataset_name.lower()}.png', dpi=300)
    plt.close()
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score'],
        'confusion_matrix': cm.tolist()
    }
    
    with open(f'model/evaluation_metrics_{dataset_name.lower()}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def plot_learning_curves(history):
    """Plot training & validation learning curves for multiple metrics"""
    metrics = ['accuracy', 'loss', 'precision', 'recall', 'auc']
    plt.figure(figsize=(20, 10))
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, idx)
        plt.plot(history.history[metric], label=f'Training {metric.capitalize()}', marker='o')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}', marker='x')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model/comprehensive_learning_curves.png', dpi=300)
    plt.close()

def train_lstm_model(data_path=None, data_df=None, seq_length=10, epochs=10, batch_size=32):
     
    os.makedirs('model', exist_ok=True)
     
    data_df, scaler, label_encoder, feature_cols, label_col = prepare_data(data_df, data_path)
     
    train_size = int(len(data_df) * 0.6)
    val_size = int(len(data_df) * 0.2)
    
    train_data = data_df[:train_size]
    val_data = data_df[train_size:train_size+val_size]
    test_data = data_df[train_size+val_size:]
    
    # Create sequences
    X_train, y_train = create_balanced_sequences(train_data, label_col, seq_length)
    X_val, y_val = create_balanced_sequences(val_data, label_col, seq_length)
    X_test, y_test = create_balanced_sequences(test_data, label_col, seq_length)
    
    # Build model
    model = build_model((seq_length, X_train.shape[2]))
    save_model_architecture(model)
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,verbose=1),
        ModelCheckpoint('model/best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        shuffle=True
    )
    
    # Evaluate and save results
    test_metrics = evaluate_model(model, X_test, y_test)
    plot_learning_curves(history)
    
    # Save model artifacts
    model.save('model/ddos_model.h5')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    joblib.dump(feature_cols, 'model/feature_columns.pkl')
    
    return model, scaler, label_encoder, feature_cols, history

if __name__ == "__main__":
    data_path = './data'
    train_lstm_model(data_path=data_path)
 