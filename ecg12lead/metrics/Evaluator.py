import math

import numpy as np
import matplotlib.pyplot as plt




class Evaluator(object):
    def __init__(self, num_classes, y_decode):
        self.num_classes = num_classes
        self.y_decode = y_decode
        self.A = np.zeros((self.num_classes, 2, 2))
        #     [TP_k FN_k]
        #     [FP_k TN_k]

    def get_accuracy(self):
        tot = self.A.sum()
        n_accurate = 0
        for j in range(self.num_classes):
            n_accurate+= np.diag(self.A[j]).sum()

        return n_accurate/tot


    def get_cm_plot(self):
        ncols = 3
        nrows = math.ceil(self.num_classes/ncols)

        fig,axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=( ncols*5, nrows*6), squeeze=False , facecolor='white')

        fig_no = 0
        for i in range(nrows): 
            for j in range(ncols):
                if (fig_no) <self.num_classes :
                    self._plot_single_confusion_matrix(cm = self.A[fig_no], axes = axes[i][j],
                                normalize    = False,
                                target_names = ['positive', 'negative'],
                                title        = "%s"%(self.y_decode[fig_no])
                                )
                    fig_no+=1
                else:
                    break
        fig.tight_layout()
        for ax in axes.flat[self.num_classes:] :
            ax.set_visible(False)
        plt.show()

        return fig

    def _update_matrix(self,target, pred):
        num_recordings = target.shape[0]
        for i in range(num_recordings):
            for j in range(self.num_classes):
                if target[i, j]==1 and pred[i, j]==1: # TP
                    self.A[j, 0, 0] += 1
                elif target[i, j]==0 and pred[i, j]==1: # FP
                    self.A[j, 1, 0] += 1
                elif target[i, j]==1 and pred[i, j]==0: # FN
                    self.A[j, 0, 1] += 1
                elif target[i, j]==0 and pred[i, j]==0: # TN
                    self.A[j, 1, 1] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')


    def add_batch(self, target, pred):
        assert target.shape == pred.shape
        target = target>0.5
        pred = pred>0.5
        self._update_matrix(target, pred)

    def reset(self):
        self.A = np.zeros((self.num_classes, 2, 2))



    def _plot_single_confusion_matrix(self, cm, axes = plt,
                            target_names=None,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):
        ''' https://www.kaggle.com/grfiv4/plot-a-confusion-matrix '''
        """
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions
        """

        accuracy = np.trace(cm) / float(np.sum(cm))
        precision = cm[0,0] /(cm[0,0]+cm[1,0]) #TP/(TP+FP)
        recall    = cm[0,0] /(cm[0,0]+cm[0,1]) #TP/(TP+FN)


        if cmap is None:
            cmap = plt.get_cmap('Blues')

        # plt.figure(figsize=(6, 4))
        axes.imshow(cm, interpolation='nearest', cmap=cmap)
        axes.title.set_text(title)
        # axes.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            axes.set_xticks(tick_marks)
            axes.set_xticklabels(target_names)
            axes.set_yticks(tick_marks)
            axes.set_yticklabels(target_names, rotation=90)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    axes.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    axes.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")


        # axes.tight_layout()
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label\naccuracy={:0.4f}; precision={:0.4f}; recall={:0.4f}'.format(accuracy,precision,recall))
        # plt.show()