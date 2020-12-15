from moanna.model.Autoencoder import *
from moanna.model.ClassificationModel import *

class Moanna(nn.Module):
    def __init__(self, input_shape, hidden_size, encoded_size, n_layers, drop_prob, fnn_hidden_size, num_classes_list, fnn_n_layers, fnn_p):
        super(Moanna, self).__init__()
        self.encoder = Encoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.decoder = Decoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.classifiers = nn.ModuleList([])
        for num_classes in num_classes_list:
            self.classifiers.append(RLModel(encoded_size, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p))
            
    def forward(self, x):
        outs = [] 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        for classifier in self.classifiers:
            outs.append(classifier(encoded))
        
        return encoded, decoded, outs 

#https://discuss.pytorch.org/t/autoencoder-and-classification-inside-the-same-model/36248
class Monna(nn.Module):
    def __init__(self, input_shape, hidden_size, encoded_size, n_layers, drop_prob, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p):
        super(Monna, self).__init__()
        self.encoder = Encoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.decoder = Decoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.classifier = RLModel(encoded_size, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = self.classifier(encoded)
        return decoded, out
    
class Monna3(nn.Module):
    def __init__(self, input_shape, hidden_size, encoded_size, n_layers, drop_prob, fnn_hidden_size, num_classes_list, fnn_n_layers, fnn_p):
        super(Monna3, self).__init__()
        #self.encoder = Encoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        #self.decoder = Decoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.classifiers = nn.ModuleList([])
        for num_classes in num_classes_list:
            self.classifiers.append(RLModel(input_shape, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p))
            
    def forward(self, x):
        outs = [] 
        #encoded = self.encoder(x)
        #decoded = self.decoder(encoded)
        for classifier in self.classifiers:
            outs.append(classifier(x))
        
        return outs    
    
    
def monna_accuracy(predictions, list_label):
    predicted_list = []
    accuracy_list = []
    
    for i in range (len(predictions)):
        prediction_softmax = torch.log_softmax(predictions[i], dim=1)
        _, predicted = torch.max(prediction_softmax,1)
        correct = torch.sum(predicted==list_label[i])
        total = list_label[i].size(0)
        accuracy = 100 * correct / total
        print(accuracy.item())
        predicted_list.append(predicted)
    
    return predicted_list

#Usage
#df = cm2df(cm, labels[mapping_file_array[2]][1])
#df.index.name = "True Label"
#df.columns.name = "Predicted Label"
def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]

#https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
#cm_analysis(y_true, predicted, "aa", lab1)    
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelweight': 'bold',
             'font.weight' : 'bold',}
    pylab.rcParams.update(params)
    
    
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'PAM50 Subtype (Class Label)'
    cm.columns.name = 'Predicted Subtype'
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=annot, annot_kws={"size":16, 'weight':'bold'}, fmt='', ax=ax, cmap="BuPu")
    plt.title("Confusion Matrix of Predicted PAM50 Subtypes", fontweight='bold')
    plt.savefig(filename)
    return cm

def monna_heatmap(y_true, y_pred, labels_mapping):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    df = cm2df(cm, labels_mapping)
    df.index.name = "True Label"
    df.columns.name = "Predicted Label"
    
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, annot_kws={"size":16}, cmap="BuPu", fmt='g')
    
    return cm
    
def monna_heatmap2(y_true, y_pred, labels_mapping, plot_out="heatmap.png"):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    return cm_analysis(y_true, y_pred, plot_out, labels_mapping) 
    
    
def monna_metrics(model, data, list_label, labels_mapping, column=2, plot_pc=True, plot_out="heatmap.png"):
    model.eval()
    encoded, decoded, predictions = model(data)
    
    predicted_list = monna_accuracy(predictions, list_label)

    pred_subtype = predicted_list[column].cpu().numpy()
    label_subtype= list_label[column].cpu().numpy()
    #monna_heatmap(label_subtype, pred_subtype)
    
    if (plot_pc):
        cm = monna_heatmap2(label_subtype, pred_subtype, labels_mapping, plot_out=plot_out)  
    else:
        cm = monna_heatmap(label_subtype, pred_subtype)
        
    return predicted_list, cm


def monna3_metrics(model, data, list_label, labels_mapping, column=2, plot_pc=True, plot_out="heatmap.png"):
    model.eval()
    predictions = model(data)
    
    predicted_list = monna_accuracy(predictions, list_label)

    pred_subtype = predicted_list[column].cpu().numpy()
    label_subtype= list_label[column].cpu().numpy()
    #monna_heatmap(label_subtype, pred_subtype)
    
    if (plot_pc):
        cm = monna_heatmap2(label_subtype, pred_subtype, labels_mapping, plot_out=plot_out)  
    else:
        cm = monna_heatmap(label_subtype, pred_subtype)
        
    return predicted_list, cm

def monna_encoded(model, data, list_label, labels_mapping, column=2, plot_pc=True, plot_out="heatmap.png"):
    model.eval()
    encoded, decoded, predictions = model(data)
        
    return encoded.detach().cpu().numpy()