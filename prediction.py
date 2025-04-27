# prediction.py
import os,json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from collections import deque
import logging
from datetime import datetime

logging.basicConfig(filename="prediction.log",level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictionModel:
    def __init__(self, model_dir='./model', sequence_length=10):
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.history = deque(maxlen=100)
        self.window = deque(maxlen=sequence_length)
        self.last_prediction_time = None
        self.prediction_threshold = 0.5
        self.load_model()
        try:
            with open('model/feature_importance.json') as fp:
                self.feature_importances = json.load(fp)
        except Exception:
            self.feature_importances = []
        self.history = []
        
        
    def load_model(self):

        logger.info("Loading TensorFlow model and artifacts...")
        try:
            model_path = os.path.join(self.model_dir, 'ddos_model.h5')
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, 'feature_columns.pkl'))
            logger.info("Model and artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Using default model configuration.")

    
    def preprocess_data(self, data_df ): 
        df = data_df.copy() 
        df.columns = [c.strip() for c in df.columns]

        drop_cols = ['Timestamp', 'Fwd Header Length']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
         
        for col in ['Source IP', 'Destination IP']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: int(hash(x) % 100000))  
        
        df = df.select_dtypes(include=['int64', 'float64']) 
        
        missing_cols = set(self.feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
         
        df = df[self.feature_columns]
         
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
         
        if self.scaler:
            try:
                scaled_data = self.scaler.transform(df)
                df = pd.DataFrame(scaled_data, columns=self.feature_columns)
            except Exception as e:
                logger.error(f"Scaling error: {str(e)}")
                raise
        
        return df

    def predict(self, traffic_data): 
        if self.model is None:
                raise RuntimeError("Model not loaded")
        current_time = datetime.now()         
        processed_data = self.preprocess_data(data_df = traffic_data)
        

         
        for _, row in processed_data.iterrows():
            self.window.append(row.values)
             
        if len(self.window) < self.sequence_length:
            return {"prediction": "Insufficient data",
                    "probability": 0.0, 
                    "status": "Insufficient data"}
             
        
        sequence = np.array([list(self.window)])
        
        try:
            adjusted_prob = float(self.model.predict(sequence, verbose=0)[0][0])
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return {
                "prediction": "Error",
                "probability": 0.0,
                "status": "Error"
            }

        pred_label = "DDoS Attack" if adjusted_prob > self.prediction_threshold else "Normal Traffic"
        status = "DDoS Attack Detected" if adjusted_prob > self.prediction_threshold else "Normal Traffic"
        
        result = {
            "timestamp": current_time.isoformat(),
            "prediction": pred_label,
            "probability": adjusted_prob,
            "status": status
        }
        
        if self.feature_importances:
            top_feature, pct = self.feature_importances[0]
        else:
            top_feature, pct = None, None
        result['top_feature'] = top_feature
        result['importance_pct'] = pct
        
        self.history.append(result)
        
        return result
        
        
    def get_history(self):         
        return list(self.history)
    
    