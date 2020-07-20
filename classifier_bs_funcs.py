import warnings
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics._classification import _check_targets, _check_zero_division, _weighted_sum, precision_recall_fscore_support



import joblib


###The real MVP###

def gen_bs_idx(y_true, n_shuffles = 10000, random_state = None):
    """
    Generate a list boostrapped indexies to generate classification metrics
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
        
    n_shuffles  :  int (default = 10,000)
        Number of bootstrapped samples to use for generation of distribution estimate.

    random_state : int or None (default None)
        Seed for reproducability.
        
    Returns
    -------
    idx_bs = 1d array-like, shape = [len(y_true, shuffles])
        Array of indicies to be included for classifier metric calculation.
    
    """
    rs = np.random.RandomState(random_state)
    
    idx_bs = rs.choice(np.arange(0, len(y_true)), [len(y_true), n_shuffles], replace = True)
    
    return(idx_bs)



###Everyone else###
def metrics_bs(metric, y_true, y_pred, ci = 95, return_raw = False, n_shuffles = 10000, n_jobs = 1, random_state = None, verbose = 0):
    """
    Generate boostrap of classification metric
    
    Parameters
    ----------
    metric: sklearn classification metric
        Metric to bootstrap 

    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted values to compare against y_true.

    ci : int (default 95)
        Confidence interval to calculate (two tailed)

    return_raw  :  bool (default False)
        Return bootstrapped classification metrics
        
    n_shuffles  :  int (default = 10,000)
        Number of bootstrapped samples to use for generation of distribution estimate.

    n_jobs  :   int (default 1)
        Number of cores to use to produce bootstrapped samples

    random_state : int or None (default None)
        Seed for reproducability.

    """

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    metric_name = str(metric).split(' ')[1]

    bs_idx = gen_bs_idx(y_true, n_shuffles, random_state)

    obs = metric(y_true, y_pred)
    results = np.array(joblib.Parallel(n_jobs = n_jobs, verbose = verbose)(joblib.delayed(metric)(y_true[n], y_pred[n]) for n in bs_idx.T))

    #Calc CI
    l = (100 - ci) / 2
    u = 100 - l

    l_ci, u_ci = np.percentile(results, [l, u])

    report_fmt = '***{}: {:.{digits}f} ({:.{digits}f} - {:.{digits}f})***'
    report = report_fmt.format(metric_name, obs, l_ci, u_ci, digits = 2)

    if return_raw:
        return report, results
    else:
        return report 




def classification_report_bs(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=2, output_dict=False,
                          zero_division="warn", ci = 95, n_shuffles = 10000, 
                          return_raw = True, n_jobs = 1, verbose = 0, random_state = None):
    """
    Build a text report showing the main classification metrics
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    digits : int
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.

    output_dict : bool (default = False)
        If True, return output as dict

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    ci : int
        Confidence interval to compute.
        Note - CI is two tailed.
        
    shuffles  :  int (default = 10,000)
        Number of bootstrapped samples to use for generation of distribution estimate.
        
    return_raw : bool (default = True)
        If True, returns an array of size n_samples predictions.
        
    

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::

            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }

        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy otherwise.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.

        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
        
     preds_bs :  1d array-like
         Predictions of estimator based on bootstrapped samples.

    """

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    labels_given = True
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)

    # labelled micro average
    micro_is_accuracy = ((y_type == 'multiclass' or y_type == 'binary') and
                         (not labels_given or
                          (set(labels) == set(unique_labels(y_true, y_pred)))))

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}"
                .format(len(labels), len(target_names))
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = ['%s' % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging

    #Compute observed values:
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight,
                                                  zero_division=zero_division)


    #Generate bootstrapped values:
    bs_idx = gen_bs_idx(y_true, n_shuffles, random_state)
    report_bs = np.array(joblib.Parallel(n_jobs = n_jobs, verbose = verbose)(joblib.delayed(precision_recall_fscore_support)(y_true[n], y_pred[n], labels=labels, average=None, sample_weight=sample_weight, zero_division=zero_division) for n in bs_idx.T))

    #Extract bs into seperate variables
    p_bs = report_bs[:, 0, :]
    r_bs = report_bs[:, 1, :]
    f1_bs = report_bs[:, 2, :]
    s_bs = report_bs[:, 3, :]

    #Calc CI
    l = (100 - ci) / 2
    u = 100 - l

    #Join obs and CI
    p_ci = np.percentile(p_bs, [l, u], axis = 0)
    p_stack = [np.hstack([p[n], np.ravel(p_ci[:, n])]) for n in range(0, len(p))]

    r_ci = np.percentile(r_bs, [l, u], axis = 0)
    r_stack = [np.hstack([r[n], np.ravel(r_ci[:, n])]) for n in range(0, len(r))]

    f1_ci = np.percentile(f1_bs, [l, u], axis = 0)
    f1_stack = [np.hstack([f1[n], np.ravel(f1_ci[:, n])]) for n in range(0, len(f1))]

    s_ci = np.percentile(s_bs, [l, u], axis = 0)
    s_stack = [np.hstack([s[n], np.ravel(s_ci[:, n])]) for n in range(0, len(s))]

    rows = zip(target_names, p_stack, r_stack, f1_stack, s_stack)

    if y_type.startswith('multilabel'):
        average_options = ('micro', 'macro', 'weighted', 'samples')
    else:
        average_options = ('micro', 'macro', 'weighted')

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers,
                                        [i for i in scores]))
    else:
        longest_last_line_heading = 'weighted avg'
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = '   {:>{width}s} ' + ' {:<20}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += '\n\n'
        row_fmt = '{:<{width}s} ' + '{:^1.{digits}f} ({:^1.{digits}f} - {:^1.{digits}f})   ' * 4 + '\n' #INCLUDE 95% CI HERE

        for row in rows:
            in_row = []
            in_row.append(row[0])
            for r in row[1:]:
                if len(r) < 2:
                    in_row.append(r)
                else:
                    for i in r:
                        in_row.append(i)
            
            
            report += row_fmt.format(*in_row, width=width, digits=digits)
        report += '\n'


    # ###IGNORE AVG ACCURACY FOR THE TIME BEING###
    # # compute all applicable averages
    # for average in average_options:
    #     if average.startswith('micro') and micro_is_accuracy:
    #         line_heading = 'accuracy'
    #     else:
    #         line_heading = average + ' avg'

    #     # compute averages with specified averaging method
    #     avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
    #         y_true, y_pred, labels=labels,
    #         average=average, sample_weight=sample_weight,
    #         zero_division=zero_division)
    #     avg = [avg_p, avg_r, avg_f1, np.sum(s)]

    #     if output_dict:
    #         report_dict[line_heading] = dict(
    #             zip(headers, [i.item() for i in avg]))
    #     else:
    #         if line_heading == 'accuracy':
    #             row_fmt_accuracy = '{:>{width}s} ' + \
    #                     ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + \
    #                     ' {:>9}\n'
    #             report += row_fmt_accuracy.format(line_heading, '', '',
    #                                               *avg[2:], width=width,
    #                                               digits=digits)
    #         else:
    #             report += row_fmt.format(line_heading, *avg,
    #                                      width=width, digits=digits)

    if output_dict:
        if 'accuracy' in report_dict.keys():
            report_dict['accuracy'] = report_dict['accuracy']['precision']
        return report_dict
    else:
        return report
