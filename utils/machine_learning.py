from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import os

class MachineLearning():

    def __init__(self, dataset_path=None):
        
        self.counter = 0
        self.flow_model = None
        self.X_flow_train = None
        self.X_flow_test = None
        self.y_flow_train = None
        self.y_flow_test = None
        self.dataset_name = None
        self.dataset_size = None
        self.model_type = None
        
        if dataset_path:
            self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path):
        print("Loading dataset ...")
        self.flow_dataset = pd.read_csv(dataset_path)
        self.dataset_name = os.path.basename(dataset_path)
        self.dataset_size = len(self.flow_dataset)
        self.preprocess_data()

    def preprocess_data(self):
        # Clean IP addresses by removing dots
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')
        
        self.X_flow = self.flow_dataset.iloc[:, :-1].values
        self.X_flow = self.X_flow.astype('float64')
        self.y_flow = self.flow_dataset.iloc[:, -1].values

        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(
            self.X_flow, self.y_flow, test_size=0.25, random_state=0)

    def train_model(self, model_type='RF'):
        if self.X_flow_train is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
        
        model_map = {
            'LR': LogisticRegression(solver='liblinear', random_state=0),
            'KNN': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
            'SVM': SVC(kernel='rbf', random_state=0),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(criterion='entropy', random_state=0),
            'RF': RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"Training {model_type} model ...")
        self.model_type = model_type
        self.classifier = model_map[model_type]
        self.flow_model = self.classifier.fit(self.X_flow_train, self.y_flow_train)
        self.evaluate_model()

    def evaluate_model(self):
        if self.flow_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        self.counter += 1
        
        y_flow_pred = self.flow_model.predict(self.X_flow_test)

        print("------------------------------------------------------------------------------")
        print("confusion matrix")
        cm = confusion_matrix(self.y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(self.y_flow_test, y_flow_pred)
        print("success accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail*100))
        print("------------------------------------------------------------------------------")
        
        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        x = ['TP','FP','FN','TN']
        x_indexes = np.arange(len(x))
        width = 0.10
        plt.xticks(ticks=x_indexes, labels=x)
        plt.title("Algorithm Results")
        plt.xlabel('Predicted Class')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        plt.style.use("default")
        
        colors = ["#1b7021", "#e46e6e", "#0000ff", "#e0d692", "#000000", "#ff6600"]
        color = colors[min(self.counter-1, len(colors)-1)]
        
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        offset = (self.counter - 1) * width
        plt.bar(x_indexes + offset, y, width=width, color=color, label=f'Model {self.counter}')
        plt.legend()
        
        if self.counter >= 5:  # Show plot after all models or when counter reaches certain point
            plt.show()

    def predict(self, data):
        if self.flow_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        if isinstance(data, pd.DataFrame):
            # Preprocess similar to training data
            data_copy = data.copy()
            data_copy.iloc[:, 2] = data_copy.iloc[:, 2].str.replace('.', '')
            data_copy.iloc[:, 3] = data_copy.iloc[:, 3].str.replace('.', '')
            data_copy.iloc[:, 5] = data_copy.iloc[:, 5].str.replace('.', '')
            X = data_copy.values.astype('float64')
        else:
            X = data
        
        return self.flow_model.predict(X)

    def save_model(self, filepath):
        if self.flow_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Create metadata dictionary
        metadata = {
            'model': self.flow_model,
            'dataset_name': self.dataset_name,
            'dataset_size': self.dataset_size,
            'model_type': self.model_type,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(metadata, filepath)
        print(f"Model saved to {filepath}")
        print(f"  - Model Type: {self.model_type}")
        print(f"  - Dataset: {self.dataset_name}")
        print(f"  - Dataset Size: {self.dataset_size} samples")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load metadata dictionary
        data = joblib.load(filepath)
        
        # Handle both old format (just model) and new format (metadata dict)
        if isinstance(data, dict) and 'model' in data:
            self.flow_model = data['model']
            self.dataset_name = data.get('dataset_name')
            self.dataset_size = data.get('dataset_size')
            self.model_type = data.get('model_type')
            print(f"Model loaded from {filepath}")
            print(f"  - Model Type: {self.model_type}")
            print(f"  - Dataset: {self.dataset_name}")
            print(f"  - Dataset Size: {self.dataset_size} samples")
            if 'training_timestamp' in data:
                print(f"  - Training Timestamp: {data['training_timestamp']}")
        else:
            # Backward compatibility: treat loaded data as the model itself
            self.flow_model = data
            print(f"Model loaded from {filepath} (legacy format)")

    # Legacy methods for backward compatibility
    def LR(self):
        self.train_model('LR')
        
    def KNN(self):
        self.train_model('KNN')
 
    def SVM(self):
        self.train_model('SVM')
        
    def NB(self):
        self.train_model('NB')
        
        
    def DT(self):
        self.train_model('DT')
        
    def RF(self):
        self.train_model('RF')
        
    def Confusion_matrix(self):
        self.evaluate_model()