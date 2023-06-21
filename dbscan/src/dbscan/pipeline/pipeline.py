from src.dbscan.model_training.model_training import model_training
from src.dbscan.model_validation.model_validation import model_validation
from src.dbscan.preprocessing.preprocessing import pre_processing
from src.utils.constant import file_path


class Dbscan:
    def __init__(self):
        """
        setting file path
         @param dataframe
        @type dataframe
        """
        self.dataset = file_path
        self.pipeline()

    def pipeline(self):
        """
        getting the dataframe and calling all the steps in building the model
        @return: accuracy
        """
        processed_dataframe = pre_processing(self.dataset)
        # trained_model = model_training(processed_dataframe)
        accuracy = model_validation(processed_dataframe)
        print(accuracy)