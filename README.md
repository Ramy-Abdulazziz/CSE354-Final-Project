# CSE354-Final-Project
Final project for CSE354

# Requirements
- Python 3.x installed
- PyTorch installed
- transformers library installed
- datasets library installed
- beautifulsoup4 library installed
- responses library installed
- scikit-learn library installed
- matplotlib library installed

## Install Requirements: 

```   
!pip install torch
!pip install transformers
!pip install datasets
!pip install beautifulsoup4
!pip install responses
!pip install scikit-learn
!pip install matplotlib
```

# Original Code

This project used a modified version of the CSE 354 HW 3 Assignment: 

## Model Class
The original DistilBERT class was modified to a general model class to load eithr a pretrained DistilBERT or ELEKTRA Model, To 
use please specifiy: 

```
model_name ='model name'
```
We had to use more than a binary classification so this was also adjusted. 
```
class Model():
  #Addappropriate model name here i.e. distilbert-base-uncased
  def __init__(self, model_name ='google/electra-small-discriminator', num_classes=3):

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

  def get_tokenizer_and_model(self):
    return self.model, self.tokenizer
```

# Dataset Loaders

There were multiple versions of the Dataset loader class that were made and modified for use with our experimentation the original base class was first modified to use our loaded Financial Phrasebook Dataset: 

```
class DatasetLoader(Dataset):

  def __init__(self, data, tokenizer):
    self.data = data
    self.tokenizer = tokenizer

  ...
```

## Dataset Loader - Unlabeled data

During our testing we wished to measure the performance of our model on new unlabeled data to this end we modified the DatasetLoader clas, specifically leveraging Pythons magic methods efficiently.

```

class DatasetLoaderSingle(Dataset):

  def __init__(self, data, tokenizer, labels=None):
      self.data = data
      self.tokenizer = tokenizer
      self.labels = labels

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      text = self.data[idx]
      inputs = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=512,
          padding='max_length',
          return_tensors='pt',
          truncation=True
      )

      input_ids = inputs['input_ids'][0]
      attention_mask = inputs['attention_mask'][0]

      if self.labels is not None:
          label = self.labels[idx]
          return input_ids, attention_mask, label
      else:
            return input_ids, attention_mask
```

## Dataset Loader - Custom Dataset

We also used another modified version that was made to accept a dictionary containing links to articles and labels. These articles were downloaded and tokenized via this class: 

```
class CustomDatasetLoader(Dataset):

    def __init__(self, article_dict, validation_dict, tokenizer):
        self.article_dict = article_dict
        self.validation_dict = validation_dict
        self.tokenizer = tokenizer

    def tokenize_data(self):
        print("Processing data..")
        tokens = []
        labels = []
        label_dict = {'positive': 2, 'negative': 0, 'neutral': 1}

        for key, value in tqdm(self.article_dict.items(), total=len(self.article_dict)):
            review = get_article_text(value)
            label = self.validation_dict[key]

            tokenized_review = self.tokenizer.encode_plus(text=review[0],
                                                          add_special_tokens=True,
                                                          max_length=512,
                                                          truncation=True,
                                                          padding='max_length',
                                                          return_tensors='pt')

            input_ids = tokenized_review['input_ids'].squeeze()

            labels.append(label_dict[label])
            tokens.append(input_ids)

        tokens = torch.stack(tokens)
        labels = torch.tensor(labels)
        dataset = TensorDataset(tokens, labels)

        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        processed_dataset = self.tokenize_data()

        data_loader = DataLoader(
            processed_dataset,
            shuffle=shuffle,
            batch_size=batch_size
        )

        return data_loader

```

# Training Class

During initial training on the Financial Phrasebank Dataset - the original Trainer class was used from HW 3 with only a slight change to handle the specifc keys in the dataset.

```
 def tokenize_data(self):
    print("Processing data..")
    tokens = []
    labels = []
    label_dict = {'positive': 2, 'negative': 0, 'neutral':1}

    sentance_list = self.data['sentence']
    label_list = self.data['label']

    ...
```

## Custom Training Class

When doing our Custom trainng we used a modified version of the original training class that was able to handle our custom data set (Yahoo Finance Articles), using our custom data loader, This involved a small modification to the initilization to use our pretrained model, and tokenizer - note the options were also modified to accomadate our needs. The execute method was also modified to use our custom data set loader: 

```

class CustomTrainer():

  def __init__(self, options):
    self.device = options['device']
    self.train_data = options['train_data']
    self.train_label = options['train_labels']
    self.val_data = options['val_data']
    self.batch_size = options['batch_size']
    self.epochs = options['epochs']
    self.save_path = options['save_path']
    self.training_type = options['training_type']
    self.model_path = options['model_path']
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    self.model.to(self.device)

    ...

    def execute(self):
    last_best = 0
    train_dataset = CustomDatasetLoader(self.train_data, self.train_label, self.tokenizer)
    train_data_loader = train_dataset.get_data_loaders(self.batch_size)
    val_dataset = DatasetLoader(self.val_data, self.tokenizer)
    val_data_loader = val_dataset.get_data_loaders(self.batch_size)
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5, eps=1e-8)
    self.set_training_parameters()
    for epoch_i in range(0, self.epochs):
        train_precision, train_recall, train_f1, train_loss = self.train(train_data_loader, optimizer)
        print(f'Epoch {epoch_i + 1}: train_loss: {train_loss:.4f} train_precision: {train_precision:.4f} train_recall: {train_recall:.4f} train_f1: {train_f1:.4f}')
        val_precision, val_recall, val_f1, val_loss = self.eval(val_data_loader)
        print(f'Epoch {epoch_i + 1}: val_loss: {val_loss:.4f} val_precision: {val_precision:.4f} val_recall: {val_recall:.4f} val_f1: {val_f1:.4f}')

        if val_f1 > last_best:
            print("Saving model..")
            self.save_transformer()
            last_best = val_f1
            print("Model saved.")
```

Also note that to use this class the method of passing options has been changed specifically variable such as save path, batch size, and epochs, are defined inside the options dictionary, rather then outside as previously implemented: 

```
trainer_options = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'train_data': article_dictionary,
    'train_labels':validation,
    'val_data': val_data, 
    'batch_size': 16,
    'epochs': 3,
    'save_path': 'models/article_trained_electra_top_2_training',
    'training_type': 'top_2_training',
    'model_path': 'models/electra-small-discriminator_top_2_training'
}

```

# Tester class

The original tester class from HW3 was modified to return more information in the form of lists - and automatically graph data from the testing process. Functions were added to add this logic: 

```
class Tester():

  ...

  def test(self, data_loader):
    self.model.eval()
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_loss = 0
    precision_list, recall_list, f1_list, loss_list = [], [], [], []

    ...
    
    return precision, recall, f1, loss, precision_list, recall_list, f1_list, loss_list
    
  def plot_metrics(self, precision_list, recall_list, f1_list, loss_list):
    plt.figure(figsize=(10, 6))

    plt.plot(precision_list, label='Precision')
    plt.plot(recall_list, label='Recall')
    plt.plot(f1_list, label='F1 Score')
    plt.plot(loss_list, label='Loss')

    plt.xlabel('Batch')
    plt.ylabel('Value')
    plt.title('Test Metrics')
    plt.legend(loc='best')
    plt.show()

  def plot_final_results(self, results, labels):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    precisions = [result[0] for result in results]
    recalls = [result[1] for result in results]
    f1_scores = [result[2] for result in results]

    x = np.arange(len(labels))
    width = 0.35

    ax[0].bar(x, precisions, width, label='Precision')
    ax[1].bar(x, recalls, width, label='Recall')
    ax[2].bar(x, f1_scores, width, label='F1-score')

    for i, metric in enumerate(['Precision', 'Recall', 'F1-score']):
        ax[i].set_ylabel(metric)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels, rotation = 90)
        ax[i].legend(loc='best')
    
    fig.suptitle('Test Performance Metrics for Different Training Regimes')
    plt.gca().yaxis.grid(False)
    plt.tight_layout()
    plt.show()

  def generate_latex_table(self, results, labels):
    header = r'''
    \begin{tabular}{|l|c|c|c|}
    \hline
    \textbf{Model and Training Regime} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} \\ \hline
    '''
    rows = []
    for label, res in zip(labels, results):
        row = f"{label} & {res[0]:.4f} & {res[1]:.4f} & {res[2]:.4f} \\\\ \\hline"
        rows.append(row)

    footer = r'''\end{tabular}'''

    table = header + '\n' + '\n'.join(rows) + '\n' + footer
    return table

  def execute(self):
    test_dataset = DatasetLoader(self.test_data, self.tokenizer)
    test_data_loader = test_dataset.get_data_loaders(self.batch_size)

    test_precision, test_recall, test_f1, test_loss, precision_list, recall_list, f1_list, loss_list = self.test(test_data_loader)

    self.plot_metrics(precision_list, recall_list, f1_list, loss_list)

    print()
    print(f'test_loss: {test_loss:.4f} test_precision: {test_precision:.4f} test_recall: {test_recall:.4f} test_f1: {test_f1:.4f}')

    return test_precision, test_recall, test_f1



```