import numpy as np
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == 'regression':
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression,
            }
        else:
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_classification,
                'MOSEI': self.__eval_mosei_classification,
                'SIMS': self.__eval_sims_classification,
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        y_pred_3 = np.argmax(y_pred, axis=1)
        mult_acc_3 = accuracy_score(y_true, y_pred_3)
        f1_score_3 = f1_score(y_true, y_pred_3, average='weighted')

        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = np.array([0 if v <= 1 else 1 for v in y_true])
        has0_acc_2 = accuracy_score(y_true_2, y_pred_2)
        has0_f1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = np.argmax(y_pred[non_zeros], axis=1)
        y_true_2 = y_true[non_zeros]
        non0_acc_2 = accuracy_score(y_true_2, y_pred_2)
        non0_f1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        return {
            'Has0_acc_2': round(has0_acc_2, 4),
            'Has0_F1_score': round(has0_f1_score, 4),
            'Non0_acc_2': round(non0_acc_2, 4),
            'Non0_F1_score': round(non0_f1_score, 4),
            'Acc_3': round(mult_acc_3, 4),
            'F1_score_3': round(f1_score_3, 4),
        }

    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __safe_corr(self, preds, truth):
        pred_std = np.std(preds)
        truth_std = np.std(truth)
        if pred_std < 1e-8 or truth_std < 1e-8:
            return 0.0
        corr = np.corrcoef(preds, truth)[0][1]
        return 0.0 if np.isnan(corr) else float(corr)

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
        test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
        test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
        test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

        mae = np.mean(np.absolute(test_preds - test_truth))
        corr = self.__safe_corr(test_preds, test_truth)
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non0_truth = (test_truth[non_zeros] > 0)
        non0_preds = (test_preds[non_zeros] > 0)
        non0_acc2 = accuracy_score(non0_truth, non0_preds)
        non0_f1 = f1_score(non0_truth, non0_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        has0_acc2 = accuracy_score(binary_truth, binary_preds)
        has0_f1 = f1_score(binary_truth, binary_preds, average='weighted')

        return {
            'Has0_acc_2': round(has0_acc2, 4),
            'Has0_F1_score': round(has0_f1, 4),
            'Non0_acc_2': round(non0_acc2, 4),
            'Non0_F1_score': round(non0_f1, 4),
            'Mult_acc_5': round(mult_a5, 4),
            'Mult_acc_7': round(mult_a7, 4),
            'MAE': round(mae, 4),
            'Corr': round(corr, 4),
        }

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1.0, a_max=1.0)
        test_truth = np.clip(test_truth, a_min=-1.0, a_max=1.0)

        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))
        corr = self.__safe_corr(test_preds, test_truth)
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        return {
            'Mult_acc_2': mult_a2,
            'Mult_acc_3': mult_a3,
            'Mult_acc_5': mult_a5,
            'F1_score': f_score,
            'MAE': mae,
            'Corr': corr,
        }

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
