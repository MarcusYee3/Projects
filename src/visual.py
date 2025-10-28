import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


def plot_saliency(model, X_input, titles):
    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    score = CategoricalScore(range(X_input.shape[0]))
    saliency_map = saliency(score, X_input)

    fig, ax = plt.subplots(2, len(titles), figsize=(12, 6))
    for i, title in enumerate(titles):
        ax[0, i].imshow(X_input[i])
        ax[0, i].set_title(title)
        ax[0, i].axis('off')
        ax[1, i].imshow(saliency_map[i], cmap='jet')
        ax[1, i].axis('off')
    plt.tight_layout()
    plt.show()