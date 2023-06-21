import pickle

from sklearn.cluster import DBSCAN

from src.utils.constant import saved_model


def model_training(data_frame):
    X_principal = data_frame[0]
    dbscan = DBSCAN(eps=1.0, min_samples=50,metric='euclidean', algorithm='auto', leaf_size=30, p=2).fit(X_principal)

    with open(saved_model/"dbscan_model.pkl", 'wb') as file:
        pickle.dump(dbscan, file)

    return dbscan