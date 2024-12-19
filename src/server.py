import numpy as np
import torch
import copy

from models import SplitLSTMAutoEncoder, DCCLSTMAutoEncoder, MLP
from torch import nn, optim
from utils import make_seq_batch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, accuracy_score

EVAL_WIN = 2000


class Server:
    def __init__(self, train_A, train_B, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_A = train_A
        self.train_B = train_B
        self.input_size_A = train_A["A"].shape[1]
        self.input_size_B = train_B["B"].shape[1]
        self.rep_size = int(config["FL"]["rep_size"])
        self.n_classes = len(set(train_A["y"]))
        self.label_modality = config["SERVER"]["label_modality"]
        self.test_modality = config["SERVER"]["test_modality"]
        self.frac = float(config["SERVER"]["frac"])
        self.n_epochs = int(config["SERVER"]["num_epochs"])
        self.lr = float(config["SERVER"]["lr"])
        self.criterion = config["SERVER"]["criterion"]
        self.optimizer = config["SERVER"]["optimizer"]
        self.model_ae = config["SIMULATION"]["model_ae"]
        self.model_sv = config["SIMULATION"]["model_sv"]
        self.mlp_dropout = 0.5 if config["SIMULATION"]["data"] == "mhealth" else 0.0

        if config["SIMULATION"]["data"] == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256

    def init_models(self):
        if self.model_ae == "split_LSTM":
            self.global_ae = SplitLSTMAutoEncoder(
                self.input_size_A, self.input_size_B, self.rep_size).double().to(self.device)

        if self.model_ae == "DCCAE_LSTM":
            self.global_ae = DCCLSTMAutoEncoder(
                self.input_size_A, self.input_size_B, self.rep_size).double().to(self.device)

        if self.model_sv == "MLP":
            self.global_sv = MLP(
                self.rep_size, self.n_classes, self.mlp_dropout).double().to(self.device)

    def select_clients(self, clients):
        """Selects clients to communicate with.

        Args:
            clients: a list of Client objects

        Returns:
            A list of selected Client objects
        """
        n_selected_clients = int(len(clients) * self.frac)
        selected_clients = np.random.choice(
            clients, n_selected_clients, replace=False)
        return selected_clients

    def average_models(self, local_models):
        """Averages local models into a new global model.

        Args:
            local_models: a list of tuples containing models, client modalities, and client weights

        Returns:
            A new global model.
        """
        w_avg_A = None
        w_avg_B = None
        n_A = 0
        n_B = 0
        for model in local_models:
            if model[1] == "A" or model[1] == "AB":
                n_A += model[2]
                if not w_avg_A:
                    w_avg_A = copy.deepcopy(model[0].state_dict())
                    for key in w_avg_A.keys():
                        if "A" in key:
                            w_avg_A[key] = w_avg_A[key] * model[2]
                else:
                    for key in w_avg_A.keys():
                        if "A" in key:
                            # multiply client weight
                            w_avg_A[key] += model[0].state_dict()[key] * \
                                model[2]

            if model[1] == "B" or model[1] == "AB":
                n_B += model[2]
                if not w_avg_B:
                    w_avg_B = copy.deepcopy(model[0].state_dict())
                    for key in w_avg_B.keys():
                        if "B" in key:
                            w_avg_B[key] = w_avg_B[key] * model[2]
                else:
                    for key in w_avg_B.keys():
                        if "B" in key:
                            w_avg_B[key] += model[0].state_dict()[key] * \
                                model[2]
        w_avg = w_avg_A if w_avg_A else w_avg_B

        if w_avg_A:
            for key in w_avg.keys():
                if "A" in key:
                    w_avg[key] = w_avg_A[key] / n_A
        if w_avg_B:
            for key in w_avg.keys():
                if "B" in key:
                    w_avg[key] = w_avg_B[key] / n_B

        return w_avg

    def train_classifier(self, label_modality, optimizer, criterion, x_train, y_train, idx_start, idx_end):
        """Trains the global classifier with labelled data on the server"""

        x = x_train[:, idx_start:idx_end, :]
        seq = torch.from_numpy(x).double().to(self.device)
        y = y_train[:, idx_start:idx_end]

        with torch.no_grad():
            rpts = self.global_ae.encode(seq, label_modality)
        targets = torch.from_numpy(y.flatten()).to(self.device)

        optimizer.zero_grad()
        output = self.global_sv(rpts)
        loss = criterion(output, targets.long())
        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == targets.view(*top_class.shape).long()
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss.item(), accuracy

    def update(self, local_models):
        """Updates the global model using received local models.

        Args:
            local_models: a list of local models

        Returns:
            A tuple containing loss and accuracy values
        """
        # Average all local models and update the global ae
        global_weights = self.average_models(local_models)
        self.global_ae.load_state_dict(global_weights)
        self.global_ae.eval()
        self.global_sv.train()

        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss().to(self.device)
        if self.optimizer == "Adam":
            optimizer = optim.Adam(self.global_sv.parameters(), lr=self.lr)

        round_loss = []
        round_accuracy = []
        for epoch in range(self.n_epochs):
            epoch_loss = []
            epoch_accuracy = []
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            x_A_train, _, y_A_train = make_seq_batch(
                self.train_A, [0], len(self.train_A["A"]), batch_size)
            _, x_B_train, y_B_train = make_seq_batch(
                self.train_B, [0], len(self.train_B["B"]), batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first

            if "A" in self.label_modality:
                seq_len = x_A_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    loss, accuracy = self.train_classifier(
                        "A", optimizer, criterion, x_A_train, y_A_train, idx_start, idx_end)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)

            if "B" in self.label_modality:
                seq_len = x_B_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    loss, accuracy = self.train_classifier(
                        "B", optimizer, criterion, x_B_train, y_B_train, idx_start, idx_end)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)

            round_loss.append(np.mean(epoch_loss))
            round_accuracy.append(np.mean(epoch_accuracy))

        return np.mean(round_loss), np.mean(round_accuracy)

    def eval(self, data_test):
        """Evaluates global models against testing data on the server.

        Args:
            data_test: a dictionary containing testing data of modalities A&B and labels y.

        Returns:
            A tuple containing loss and accuracy values
        """
        self.global_ae.eval()
        self.global_sv.eval()
        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss().to(self.device)

        if self.test_modality == "A":
            x_samples = np.expand_dims(data_test["A"], axis=0)
        elif self.test_modality == "B":
            x_samples = np.expand_dims(data_test["B"], axis=0)
        y_samples = np.expand_dims(data_test["y"], axis=0)

        win_loss = []
        win_accuracy = []
        win_f1 = []
        win_per_class_f1 = [] 
        n_samples = x_samples.shape[1]
        n_eval_process = n_samples // EVAL_WIN + 1
        num_classes = y_samples.shape[1]

        for i in range(n_eval_process):
            idx_start = i * EVAL_WIN
            idx_end = np.min((n_samples, idx_start+EVAL_WIN))
            x = x_samples[:, idx_start:idx_end, :]
            y = y_samples[:, idx_start:idx_end]

            inputs = torch.from_numpy(x).double().to(self.device)
            targets = torch.from_numpy(y.flatten()).to(self.device)
            rpts = self.global_ae.encode(inputs, self.test_modality)
            output = self.global_sv(rpts)

            loss = criterion(output, targets.long())
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            np_gt = y.flatten()
            np_pred = top_class.squeeze().cpu().detach().numpy()

            # Calculate the weighted F1 score
            weighted_f1 = f1_score(np_gt, np_pred, average="weighted")

            # Calculate per-class F1 scores, parameter average=None gives per class score
            per_class_f1 = f1_score(np_gt, np_pred, average=None)
            #  # Ensure win_per_class_f1 has the correct shape
            # if len(win_per_class_f1) == 0:
            #     win_per_class_f1 = [[[] for _ in range(num_classes)]]

            # # Append per-class F1 scores for the current batch
            # for c in range(num_classes):
            #     win_per_class_f1[0][c].append(per_class_f1[c])

            win_per_class_f1.append(per_class_f1)  # Append per-class F1 scores

            win_loss.append(loss.item())
            win_accuracy.append(accuracy)
            win_f1.append(weighted_f1)
      
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
              # Calculate and print the mean per-class F1 scores
            # mean_per_class_f1 = np.mean(win_per_class_f1, axis=0)
            for i, f1 in enumerate(per_class_f1):
                print(f"Class {i}: F1-score = {f1:.4f}")

        return np.mean(win_loss), np.mean(win_accuracy), np.mean(win_f1)

    def eval_per_class(self, data_test):
        """
        Evaluates global models against testing data with detailed metrics.
        
        Args:
            data_test: Dictionary containing testing data of modalities A&B and labels y.
            
        Returns:
            Tuple containing overall and per-class metrics
        """
        self.global_ae.eval()
        self.global_sv.eval()
        
        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Prepare input data
        if self.test_modality == "A":
            x_samples = np.expand_dims(data_test["A"], axis=0)
        elif self.test_modality == "B":
            x_samples = np.expand_dims(data_test["B"], axis=0)
        y_samples = np.expand_dims(data_test["y"], axis=0)
        
        # Initialize metric collectors
        all_predictions = []
        all_targets = []
        win_loss = []
        n_samples = x_samples.shape[1]
        n_eval_process = n_samples // EVAL_WIN + 1
        
        # Evaluate in windows
        for i in range(n_eval_process):
            idx_start = i * EVAL_WIN
            idx_end = np.min((n_samples, idx_start+EVAL_WIN))
            x = x_samples[:, idx_start:idx_end, :]
            y = y_samples[:, idx_start:idx_end]
            
            inputs = torch.from_numpy(x).double().to(self.device)
            targets = torch.from_numpy(y.flatten()).to(self.device)
            
            # Get model predictions
            rpts = self.global_ae.encode(inputs, self.test_modality)
            output = self.global_sv(rpts)
            
            # Calculate loss
            loss = criterion(output, targets.long())
            win_loss.append(loss.item())
            
            # Store predictions and targets
            _, predictions = output.max(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_idx in range(self.n_classes):
            class_mask = all_targets == class_idx
            if np.any(class_mask):
                class_predictions = all_predictions[class_mask]
                class_targets = all_targets[class_mask]
                
                class_correct = class_predictions == class_targets
                class_metrics[class_idx] = {
                    'accuracy': np.mean(class_correct),
                    'support': np.sum(class_mask),
                    'weighted_f1': f1_score(class_targets, class_predictions, average='weighted'),
                    'macro_f1': f1_score(class_targets, class_predictions, average='macro')
                }
        
        # Calculate overall metrics
        overall_metrics = {
            'loss': np.mean(win_loss),
            'accuracy': accuracy_score(all_targets, all_predictions),
            'weighted_f1': f1_score(all_targets, all_predictions, average='weighted'),
            'macro_f1': f1_score(all_targets, all_predictions, average='macro'),
            'confusion_matrix': confusion_matrix(all_targets, all_predictions)
        }
        
        # Print detailed evaluation results
        # print("\nEvaluation Results:")
        # print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
        # print(f"Weighted F1: {overall_metrics['weighted_f1']:.4f}")
        # print(f"Macro F1: {overall_metrics['macro_f1']:.4f}")
        print("\nPer-class metrics:")
        for class_idx, metrics in class_metrics.items():
             print(f"\nClass {class_idx}:")
             print(f"  Accuracy: {metrics['accuracy']:.4f}")
             print(f"  Weighted_f1: {metrics['weighted_f1']:.4f}") 
             print(f"  Macro_f1: {metrics['macro_f1']:.4f}")
             print(f"  Support: {metrics['support']}")
        
        # print("\nConfusion Matrix:")
        # print(overall_metrics['confusion_matrix'])
        
        # Return both summary metrics and detailed metrics
        return (
            overall_metrics['loss'],
            overall_metrics['accuracy'],
            overall_metrics['weighted_f1'],
            overall_metrics['macro_f1'],
            class_metrics
        )