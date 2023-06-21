import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def pre_processing(data_frame):
    """
        getting the dataframe and pre processing it
        @param data_frame
        @return dataframe
    """
    read_file = pd.read_csv(data_frame, delimiter=";",
                            low_memory=False)
    data_frame = pd.DataFrame(read_file)
    data_frame = data_frame.sample(n=100000)
    processed_dropped_data = drop_columns(data_frame)
    missing_value_data = handling_missing_values(processed_dropped_data)
    selected_feature_data = feature_selection(missing_value_data)
    scaled_data = scaling(selected_feature_data)
    normalized_data = pd.DataFrame(scaled_data)
    new_reduced_data = dimensionality_reduction(normalized_data)
    data = [new_reduced_data, selected_feature_data]
    return data


def dimensionality_reduction(data_frame):
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(data_frame)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    return X_principal


def drop_columns(data_frame):
    """
    getting the dataframe and dropping the un_necessary column
    @param data_frame
    @return data_frame
    """
    dropping_columns = ['Date', 'Time']
    data_frame = data_frame.drop(dropping_columns, axis=1)
    return data_frame


def handling_missing_values(data_frame):
    """
    getting the dataframe and handling missing values in each column
    :param data_frame
    :return data_frame
    """
    columns_to_preprocess = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                             'Sub_metering_1',
                             'Sub_metering_2', 'Sub_metering_3']
    for column in columns_to_preprocess:
        data_frame[column] = data_frame[column].replace('?', np.nan)
        data_frame[column] = data_frame[column].astype(float)
        mean_value = data_frame[column].mean()
        data_frame[column].fillna(mean_value, inplace=True)
    return data_frame


def feature_selection(data_frame):
    """
    getting the dataframe and doing feature selection
    :param data_frame:
    :return data_frame
    """
    selected_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    feature_dataframe = data_frame[selected_features]
    return feature_dataframe


def scaling(data_frame):
    """
    getting the dataframe and normalizing it
    @param data_frame
    @return data_frame
    """
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data_frame)

    return scaled_data
