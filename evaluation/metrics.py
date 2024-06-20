import numpy as np
import pandas as pd
import torch
import os
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman


def save_hr_outputs(gt_hr_fft_all, gt_hr_peak_all, predict_hr_fft_all, predict_hr_peak_all, config):
    output_dir = config.TEST.OUTPUT_SAVE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Filename ID to be used in any output files that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    column_name = 'gt_hr_fft', 'gt_hr_peak', 'predict_hr_fft', 'predict_hr_peak'

    file_name = '_gt_predict.csv'

    output_path = os.path.join(output_dir, filename_id + file_name)

    pd.DataFrame(data=np.array([gt_hr_fft_all, gt_hr_peak_all, predict_hr_fft_all, predict_hr_peak_all]).T,
                 columns=column_name).to_csv(output_path, index=False)

    print('')
    print('Saving hr outputs to:', output_path)


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = torch.stack([v for k, v in sorted(data.items(), key=lambda x: x[0])])

    if flatten:
        sort_data = np.array(torch.squeeze(sort_data.cpu()))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_FFT = list()
    SNR_PEAK = list()
    MACC_all = list()
    print("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        gt_hrs = np.load(config.TEST.DATA.CACHED_PATH + os.sep + "{0}_raw_hrs.npy".format(index)) if os.path.exists(
            config.TEST.DATA.CACHED_PATH + os.sep + "{0}_raw_hrs.npy".format(index)) else None
        chunk_length = prediction.shape[1]

        if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Unsupported label type in testing!")

        for i, (pred_window, label_window) in enumerate(zip(prediction, label)):
            gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
            gt_hr_peak_all.append(gt_hr_peak) if gt_hrs is None else gt_hr_peak_all.append(gt_hrs[i + chunk_length - 1])
            predict_hr_peak_all.append(pred_hr_peak)
            SNR_PEAK.append(SNR)

            gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
            gt_hr_fft_all.append(gt_hr_fft) if gt_hrs is None else gt_hr_fft_all.append(gt_hrs[i + chunk_length - 1])
            predict_hr_fft_all.append(pred_hr_fft)
            SNR_FFT.append(SNR)
            MACC_all.append(macc)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    gt_hr_fft_all = np.array(gt_hr_fft_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    predict_hr_peak_all = np.array(predict_hr_peak_all)

    save_hr_outputs(gt_hr_fft_all, gt_hr_peak_all, predict_hr_fft_all, predict_hr_peak_all, config)

    SNR_FFT = np.array(SNR_FFT)
    SNR_PEAK = np.array(SNR_PEAK)
    MACC_all = np.array(MACC_all)
    num_test_samples = len(predict_hr_fft_all)
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            print("===MAE===")
            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
            print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
            standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
            print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
        elif metric == "RMSE":
            print("===RMSE===")
            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
            standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
            print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
            standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
            print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
        elif metric == "MAPE":
            print("===MAPE===")
            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(
                num_test_samples) * 100
            print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(
                num_test_samples) * 100
            print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
        elif metric == "Pearson":
            print("===Pearson===")
            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            correlation_coefficient = Pearson_FFT[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
            print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
            correlation_coefficient = Pearson_PEAK[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
            print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
        elif metric == "SNR":
            print("===SNR===")
            standard_error = np.std(SNR_FFT) / np.sqrt(num_test_samples)
            print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(np.mean(SNR_FFT), standard_error))
            standard_error = np.std(SNR_PEAK) / np.sqrt(num_test_samples)
            print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(np.mean(SNR_PEAK), standard_error))
        elif metric == "MACC":
            print("===MACC===")
            MACC_avg = np.mean(MACC_all)
            standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
            print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
        elif "AU" in metric:
            pass
        elif "BA" in metric:
            compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
            compare.scatter_plot(
                x_label='GT PPG HR [bpm]',
                y_label='rPPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
            compare.difference_plot(
                x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                y_label='Average of rPPG HR and GT PPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')

            compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
            compare.scatter_plot(
                x_label='GT PPG HR [bpm]',
                y_label='rPPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
            compare.difference_plot(
                x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                y_label='Average of rPPG HR and GT PPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
        else:
            raise ValueError("Wrong Test Metric Type")
