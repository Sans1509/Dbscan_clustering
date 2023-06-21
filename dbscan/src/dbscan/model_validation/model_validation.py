import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.utils.constant import model_location

dbscan_model = pickle.load(open(model_location, 'rb'))


def model_validation(data_frame):
    X_principal = data_frame[0]
    new_data_frame = data_frame[1]
    labels = dbscan_model.labels_

    unique_labels = set(labels)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
    colormap = {label: color for label, color in zip(unique_labels, colors)}
    unknown_label_color = 'black'

    cvec = [colormap.get(label, unknown_label_color) for label in labels]

    plt.figure(figsize=(9, 9))
    plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap.get(label, unknown_label_color),
                   markersize=8, label=f'Label {label}') for label in unique_labels]
    plt.legend(handles=legend_elements)

    plt.xlabel('P1')
    plt.ylabel('P2')
    plt.title('DBSCAN Clustering')

    plt.show()

    silhouette_avg = silhouette_score(new_data_frame, labels)

    return silhouette_avg
