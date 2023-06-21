import unittest

from src.dbscan.preprocessing.preprocessing import drop_columns
from src.utils.constant import file_path
import pandas as pd
import logging


class Testing_Dbscan:
    def __init__(self):
        """
        setting the file path
         @param dataframe
        @type dataframe
        """
        self.dataset = file_path

    def test_pre_processing(self):
        """
            getting the dataframe and pre processing it
            @param data_frame
            @return dataframe
        """
        read_file = pd.read_csv(self.dataset, delimiter=";",
                                low_memory=False)
        data_frame = pd.DataFrame(read_file)
        data_frame = data_frame.sample(n=10000)
        processed_dropped_data = drop_columns(data_frame)
        if isinstance(data_frame, pd.DataFrame):
            logging.info("This is a Dataframe")
        else:
            logging.info("This is not Dataframe")
        return processed_dropped_data

    def test_drop_columns(self):
        """
        getting the dataframe and dropping the un_necessary column
        @param data_frame
        @return data_frame
        """
        read_file = pd.read_csv(self.dataset, delimiter=";",
                                low_memory=False)
        data_frame = pd.DataFrame(read_file)
        data_frame = data_frame.sample(n=10000)
        dropping_columns = ['Date', 'Time']
        new_data_frame = data_frame.drop(dropping_columns, axis=1)
        if len(new_data_frame) == 10000:
            logging.info("The Dataframe is in right length")
        else:
            logging.info("The Dataframe is in wrong length")
        return new_data_frame


class Test_class(unittest.TestCase):
    def test_for_dataframe(self):
        with self.assertLogs() as captured:
            df_check = Testing_Dbscan()
            df_check.test_pre_processing()
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "This is a Dataframe")

    def test_for_length_of_dataframe(self):
        with self.assertLogs() as captured:
            divide_check = Testing_Dbscan()
            divide_check.test_drop_columns()
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "The Dataframe is in right length")
