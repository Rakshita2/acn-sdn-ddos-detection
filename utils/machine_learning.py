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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse
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

    def train_model(self, model_type='RF', n_jobs=10):
        if self.X_flow_train is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
        
        model_map = {
            'LR': LogisticRegression(solver='liblinear', random_state=0, verbose=1, n_jobs=n_jobs),
            'KNN': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=n_jobs),
            'SVM': SVC(kernel='rbf', random_state=0, verbose=1),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(criterion='entropy', random_state=0),
            'RF': RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0, verbose=1, n_jobs=n_jobs),
            'GB': GradientBoostingClassifier(n_estimators=10, random_state=0, verbose=1),
            'AB': AdaBoostClassifier(n_estimators=10, random_state=0),
            'BC': BaggingClassifier(n_estimators=10, random_state=0, verbose=1, n_jobs=n_jobs),
            'VC': VotingClassifier(estimators=[
                ('lr', LogisticRegression(solver='liblinear', random_state=0, verbose=1, n_jobs=n_jobs)),
                ('dt', DecisionTreeClassifier(criterion='entropy', random_state=0)),
                ('rf', RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0, verbose=1, n_jobs=n_jobs))
            ], voting='hard')
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
        
    def Confusion_matrix(self):
        self.evaluate_model()

    def evaluate_all_models(self, model_types=None, show_confusion_grid=True):
        """
        Train and evaluate a set of models, collect metrics (accuracy, sensitivity, specificity),
        and show comparison plots.

        model_types: list of model codes to evaluate (e.g. ['LR','KNN','SVM','NB','DT','RF','GB','AB','BC','VC']).
        show_confusion_grid: if True, show a grid of confusion matrices for each model.

        Assumes a binary classification problem (labels 0 and 1). For multiclass problems,
        sensitivity/specificity are computed only for the positive class (label 1).
        """
        if self.X_flow_train is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        if model_types is None:
            model_types = ['LR', 'KNN', 'SVM', 'NB', 'DT', 'RF', 'GB', 'AB', 'BC', 'VC']

        results = []

        # Reset internal counter used by the existing plotting helper
        self.counter = 0

        for m in model_types:
            try:
                # train_model will set self.flow_model and call evaluate_model (prints + incremental bar plot)
                self.train_model(m)
            except Exception as e:
                print(f"Skipping model {m} due to error: {e}")
                continue

            # Predict on test set and compute metrics
            y_pred = self.flow_model.predict(self.X_flow_test)
            cm = confusion_matrix(self.y_flow_test, y_pred)

            # Interpret cm for binary labels. If not binary, attempt best-effort handling.
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
            else:
                # Attempt to map assuming labels are sorted; fallback to zeros
                try:
                    # convert to 2x2 by summing other classes into an 'other' bucket (not ideal)
                    TN = cm[0, 0]
                    TP = cm[-1, -1]
                    FP = cm[0, -1] if cm.shape[1] > 1 else 0
                    FN = cm[-1, 0] if cm.shape[0] > 1 else 0
                except Exception:
                    TN = FP = FN = TP = 0

            accuracy = accuracy_score(self.y_flow_test, y_pred)
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

            results.append({
                'model': m,
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
            })

        if len(results) == 0:
            print("No models evaluated.")
            return results

        # Plot comparison of accuracy / sensitivity / specificity
        labels = [r['model'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        sensitivities = [r['sensitivity'] for r in results]
        specificities = [r['specificity'] for r in results]

        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, accuracies, width, label='Accuracy', color='#4c72b0')
        plt.bar(x, sensitivities, width, label='Sensitivity', color='#55a868')
        plt.bar(x + width, specificities, width, label='Specificity', color='#c44e52')
        plt.xticks(ticks=x, labels=labels)
        plt.ylim(0, 1.0)
        plt.ylabel('Score')
        plt.title('Model comparison: Accuracy / Sensitivity / Specificity')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Optional: show confusion matrices in a grid for visual comparison
        if show_confusion_grid:
            n = len(results)
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.array([axes])

            for idx, res in enumerate(results):
                r = idx // cols
                c = idx % cols
                ax = axes[r, c]
                cm = res['confusion_matrix']
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title(res['model'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                # annotate
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(int(cm[i, j]), 'd'), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2. else 'black')

            # hide any unused subplots
            total_plots = rows * cols
            for idx in range(n, total_plots):
                r = idx // cols
                c = idx % cols
                axes[r, c].axis('off')

            plt.tight_layout()
            plt.show()

        return results


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate multiple ML models on a flow dataset')
    parser.add_argument('-d', '--dataset', default='../dataset/FlowStatsfile.csv', help='Path to the CSV dataset')
    parser.add_argument('-m', '--models', nargs='*', help='List of model codes to evaluate, e.g. LR KNN SVM NB DT RF')
    parser.add_argument('--no-confusion-grid', dest='confusion_grid', action='store_false', help='Do not show confusion matrix grid')
    args = parser.parse_args()

    ml = MachineLearning(args.dataset)
    model_list = args.models if args.models else None
    ml.evaluate_all_models(model_list, show_confusion_grid=args.confusion_grid)


if __name__ == '__main__':
    main()