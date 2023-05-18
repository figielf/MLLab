import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

from tests.utils.timeseries_data_utils import get_airline_passengers_timeseries_data


def rmse(y, t):
    return np.sqrt(np.mean((y - t) ** 2))


def mae(y, t):
    return np.mean(np.abs(y - t))


def calc_scores(train_predictions, test_predictions):
    return (rmse(train['Passengers'], train_predictions),
            rmse(test['Passengers'], test_predictions),
            mae(train['Passengers'], train_predictions),
            mae(test['Passengers'], test_predictions))


def exponential_smoothing_1d(x, alpha):
    assert len(x.shape) == 1
    result = np.zeros(x.shape)
    result[0] = x[0]
    for i, xi in enumerate(x[1:]):
        result[i + 1] = alpha * xi + (1 - alpha) * result[i]
    return result


if __name__ == '__main__':
    ALPHA = 0.2
    N_TEST = 12

    data, _ = get_airline_passengers_timeseries_data(train_size=1.0, should_plot=False)
    train = data.iloc[:-N_TEST]
    test = data.iloc[-N_TEST:]

    train_idx = data.index <= train.index[-1]
    test_idx = data.index > train.index[-1]

    # exponential smoothing
    data.loc[train_idx, 'EWMSmoothed'] = train['Passengers'].ewm(alpha=ALPHA, adjust=False).mean()  # ewm calculates smoothed data so no need to shift data
    exponential_smoothing_check = exponential_smoothing_1d(train['Passengers'].to_numpy(), alpha=ALPHA)  # this function calculates smoothed data so no need to shift data
    assert np.allclose(data.loc[train_idx, 'EWMSmoothed'], exponential_smoothing_check)

    # simple exponential smoothing without trend and seasonality
    ses = SimpleExpSmoothing(train['Passengers'], initialization_method='legacy-heuristic')
    res = ses.fit(smoothing_level=ALPHA, optimized=False)
    data.loc[train_idx, 'SESSmoothed'] = res.fittedvalues.shift(-1)  # fittedvalues are one step forecasts so need to shift data one step left
    data.loc[train_idx, 'SESPredicion'] = res.fittedvalues
    assert np.allclose(res.fittedvalues, res.predict(start=train.index[0], end=train.index[-1]))

    data.loc[test_idx, 'SESPredicion'] = res.forecast(N_TEST)
    assert np.allclose(data.loc[test_idx, 'SESPredicion'], res.predict(start=test.index[0], end=test.index[-1]))

    ses_estimated = SimpleExpSmoothing(train['Passengers'], initialization_method='legacy-heuristic')
    ses_estimated = ses_estimated.fit()
    data.loc[train_idx, 'SESestPredicion'] = ses_estimated.fittedvalues
    data.loc[test_idx, 'SESestPredicion'] = ses_estimated.forecast(N_TEST)

    # exponential smoothing with trend but without seasonality
    holt = Holt(data['Passengers'], initialization_method='legacy-heuristic')
    holt = holt.fit()
    data.loc[train_idx, 'HoltPredicion'] = holt.fittedvalues
    data.loc[test_idx, 'HoltPredicion'] = holt.forecast(N_TEST)

    # exponential smoothing with trend and seasonality
    holt_winters = ExponentialSmoothing(train['Passengers'], initialization_method='legacy-heuristic', trend='add', seasonal='add', seasonal_periods=12)
    holt_winters = holt_winters.fit()
    data.loc[train_idx, 'HoltWintersPredicion'] = holt_winters.fittedvalues
    data.loc[test_idx, 'HoltWintersPredicion'] = holt_winters.forecast(N_TEST)

    holt_winters_seasonal_mul = ExponentialSmoothing(train['Passengers'], initialization_method='legacy-heuristic', trend='add', seasonal='mul', seasonal_periods=12)
    holt_winters_seasonal_mul = holt_winters_seasonal_mul.fit()
    data.loc[train_idx, 'HoltWinters_seasonal_mul_Predicion'] = holt_winters_seasonal_mul.fittedvalues
    data.loc[test_idx, 'HoltWinters_seasonal_mul_Predicion'] = holt_winters_seasonal_mul.forecast(N_TEST)

    holt_winters_all_mul = ExponentialSmoothing(train['Passengers'], initialization_method='legacy-heuristic', trend='mul', seasonal='mul', seasonal_periods=12)
    holt_winters_all_mul = holt_winters_all_mul.fit()
    data.loc[train_idx, 'HoltWinters_all_mul_Predicion'] = holt_winters_all_mul.fittedvalues
    data.loc[test_idx, 'HoltWinters_all_mul_Predicion'] = holt_winters_all_mul.forecast(N_TEST)

    # plot values
    plt.figure(figsize=(20, 10))

    plt.subplot(3, 2, 1)
    plt.plot(data[['Passengers']], label='Passengers')
    plt.plot(data[['SESPredicion']], label=f'SES Prediction (alpha={ALPHA})')
    plt.legend()
    train_rmse, test_rmse, train_mae, test_mae = calc_scores(train_predictions=data.loc[train_idx, 'SESPredicion'], test_predictions=data.loc[test_idx, 'SESPredicion'])
    plt.title(f'One step forecast (train: (rmse={train_rmse:.2f},  mae={train_mae:.2f}), test: (rmse={test_rmse:.2f}), mae={test_mae:.2f}))')

    plt.subplot(3, 2, 2)
    plt.plot(data[['Passengers']], label='Passengers')
    plt.plot(data[['SESestPredicion']], label=f'SES Prediction (estimated alpha={ses_estimated.params["smoothing_level"]})')
    plt.legend()
    train_rmse, test_rmse, train_mae, test_mae = calc_scores(train_predictions=data.loc[train_idx, 'SESestPredicion'], test_predictions=data.loc[test_idx, 'SESestPredicion'])
    plt.title(f'One step forecast (train: (rmse={train_rmse:.2f},  mae={train_mae:.2f}), test: (rmse={test_rmse:.2f}), mae={test_mae:.2f}))')

    plt.subplot(3, 2, 3)
    plt.plot(data[['Passengers']], label='Passengers')
    plt.plot(data[['HoltPredicion']], label=f'Holt Prediction (estimated alpha={holt.params["smoothing_level"]})')
    plt.legend()
    train_rmse, test_rmse, train_mae, test_mae = calc_scores(train_predictions=data.loc[train_idx, 'HoltPredicion'], test_predictions=data.loc[test_idx, 'HoltPredicion'])
    plt.title(f'One step forecast (train: (rmse={train_rmse:.2f},  mae={train_mae:.2f}), test: (rmse={test_rmse:.2f}), mae={test_mae:.2f}))')

    plt.subplot(3, 2, 4)
    plt.plot(data[['Passengers']], label='Passengers')
    plt.plot(data[['HoltWintersPredicion']], label=f'Holt Winters Prediction (estimated alpha={holt_winters.params["smoothing_level"]})')
    plt.legend()
    train_rmse, test_rmse, train_mae, test_mae = calc_scores(train_predictions=data.loc[train_idx, 'HoltWintersPredicion'], test_predictions=data.loc[test_idx, 'HoltWintersPredicion'])
    plt.title(f'One step forecast (train: (rmse={train_rmse:.2f},  mae={train_mae:.2f}), test: (rmse={test_rmse:.2f}), mae={test_mae:.2f}))')

    plt.subplot(3, 2, 5)
    plt.plot(data[['Passengers']], label='Passengers')
    plt.plot(data[['HoltWinters_seasonal_mul_Predicion']], label=f'Holt Winters Seasonal mul Prediction (estimated alpha={holt_winters.params["smoothing_level"]})')
    plt.legend()
    train_rmse, test_rmse, train_mae, test_mae = calc_scores(train_predictions=data.loc[train_idx, 'HoltWinters_seasonal_mul_Predicion'], test_predictions=data.loc[test_idx, 'HoltWinters_seasonal_mul_Predicion'])
    plt.title(f'One step forecast (train: (rmse={train_rmse:.2f},  mae={train_mae:.2f}), test: (rmse={test_rmse:.2f}), mae={test_mae:.2f}))')

    plt.subplot(3, 2, 6)
    plt.plot(data[['Passengers']], label='Passengers')
    plt.plot(data[['HoltWinters_all_mul_Predicion']], label=f'Holt Winters both mul Prediction (estimated alpha={holt_winters.params["smoothing_level"]})')
    plt.legend()
    train_rmse, test_rmse, train_mae, test_mae = calc_scores(train_predictions=data.loc[train_idx, 'HoltWinters_all_mul_Predicion'], test_predictions=data.loc[test_idx, 'HoltWinters_all_mul_Predicion'])
    plt.title(f'One step forecast (train: (rmse={train_rmse:.2f},  mae={train_mae:.2f}), test: (rmse={test_rmse:.2f}), mae={test_mae:.2f}))')

    plt.show()
