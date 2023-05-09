# CSE354-Final-Project
Final project for CSE354

# Group Members

- Ramy Abdulazziz
- Ryan Engel
- Sahibjot Bhullar

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
# How to run
## Loading the Dataset
After installing and importing requirments, for the first portion of our project we train on the Financial Phrasebank Dataset, we use different splits of the set to make a training, testing, and validation sets.
## Financial Phrasebank
```
train_data = load_dataset('financial_phrasebank', 'sentences_75agree', split='train')
val_data = load_dataset('financial_phrasebank', 'sentences_66agree', split='train')
test_data = load_dataset('financial_phrasebank', 'sentences_allagree', split='train')
```
## Training
Be sure to fill out the save path you would like to use here along with the batch size and epochs: 

```
BATCH_SIZE = 16
EPOCHS = 3
SAVE_PATH = 'add your save path'
```
and be sure to navigate to the appropriate directory that is defined for your save path we assume a google drive folder will be mounted: 
```
from google.colab import drive
drive.mount('/content/drive')

%cd "drive/MyDrive/"
```
To do the initial training with this dataset, use the Trainer() class similar to HW3: 


```
options = {}
options['batch_size'] = BATCH_SIZE
options['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
options['train_data'] = train_data
options['val_data'] = val_data
options['save_path'] = SAVE_PATH + '_all_training'
options['epochs'] = EPOCHS
options['training_type'] = 'all_layers'
trainer = Trainer(options)
trainer.execute()
```

and be sure to define the appropriate model name in the Model class before running (i.e. distilbert-base-uncased): 

```
class Model():
  #Addappropriate model name here i.e. distilbert-base-uncased
  def __init__(self, model_name ='google/electra-small-discriminator', num_classes=3):

    ...
```

Also be sure that the dataloader class is loaded appropriately before begining training.

## Testing
For initial testing first be sure to initialize the results and labels lists: 

```
results = []
labels = []
```
Then similar to HW3 be sure to define the appropriate save path in the Tester options, and run the test.
```
options = {}
options['batch_size'] = BATCH_SIZE
options['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
options['test_data'] = test_data
options['save_path'] = 'models/your model path'
tester = Tester(options)
D2_results = tester.execute()

```
The execute function here is modified to return a list of results, and automatically provide a graph of test metrics that is visualized after the test proceure is completed.

After running the tests you can generate a graph to compare all model performance results by calling the appropriate function shown below, optionally a Latex table can also be generated: 

```
results.clear()
labels.clear()
results.append(D4_results)
labels.append('DistilBERT top 4')
...
...
tester.plot_final_results(results, labels)
tester.generate_latex_table(results,labels)
```


## Few Shot Testing 

To test models on unlabeled financial articles from Yahoo Finance, instantiate the article_dictionary and validation dictionary: 

```
article_dictionary = {}
validation = {}
```

Then define a variable to hold to article url, add it to the article_dictionary, and add the label to the validation dictionary. The get_article_text() function currently is made to work specifically with Yahoo Finance Articles, but the model can theoretically be tested using any article link, as long as a proper DOM traversal is defined in the function, to retrieve the nececarry text: 

```
negative_warren_buffet= r'https://finance.yahoo.com/news/buffett-on-the-regional-bank-crisis-messed-up-incentives-and-poor-communication-211138275.html'
article_dictionary['warren buffet negative'] = negative_warren_buffet
validation ['warren buffet negative'] = 'negative'
```

Next, instantiate the ModelEvaluator class with the appropriate keyword arguments. Note that for ELECTRA models, the keyword argument model_type = 'electra' MUST be passed. This is not nececarry for DistilBERT models. Be sure to pass in the appropriate model path: 

```
evaluator = ModelEvaluator(model_path='models/path to your model', model_type='electra')

```

The prediction sequence runs on a loop, passing each url to the evaluator, and storing its predictions and metrics. 

```
e2_preds = {}
for description, link in article_dictionary.items():
  predictions = evaluator.analyze_article(link)
  e2_preds[description] = predictions[0]

  print(f'\n The article {description} is predicted to be {predictions[0]}')
e2_metrics = evaluator.calculate_metrics(list(e2_preds.values()), list(validation.values()))
print(f'Accuracy:{e2_metrics[0]:.4f} Precision:{e2_metrics[1]:.4f} Recall:{e2_metrics[2]:.4f} f1:{e2_metrics[3]:.4f}')
```

Then run the appropriate code blocks to graph and visualize the results. An optional latex table can also be generated. 
## Custom Training on Financial Articles
To do custom training on financial articles, first run the code blocks defining the custom trainer and data loaders, and then appropriately define the trainer_options. Be sure to pass the appropriate save path, training type, and model path: 

```
trainer_options = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'train_data': article_dictionary,
    'train_labels':validation,
    'val_data': val_data, 
    'batch_size': 16,
    'epochs': 3,
    'save_path': 'models/your save path here',
    'training_type': 'training type',
    'model_path': 'models/your model path here'
}

```
Then run the trainer:

```
trainer = CustomTrainer(trainer_options)
trainer.execute()
```

To test the newly trained model, the process is the same as the previously outlined Few Shot testing. 

# Yahoo Finance Articles Used
Here we provide the finance articles used to test and train our models: 


[negative_interest_rate](https://finance.yahoo.com/news/federal-reserve-interest-rate-decision-may-3-155524134.html)

[housing_market_positive](https://finance.yahoo.com/news/housing-confidence-jumps-by-largest-amount-in-two-years-183912533.html)


[stocks_neutral](https://finance.yahoo.com/news/stock-market-news-today-live-updates-may-8-115018101.html)


[tesla_positive](https://finance.yahoo.com/news/tesla-stock-jumps-55-on-friday-snaps-longest-weekly-losing-streak-since-2021-201813547.html)


[lyft_negative](https://finance.yahoo.com/news/lyft-q1-earnings-143301253.html)


[peloton_negative](https://finance.yahoo.com/news/peloton-stock-tanks-on-forecast-for-challenging-fourth-quarter-194437985.html)


[ai_neutral](https://finance.yahoo.com/news/ai-is-an-inevitability-but-theres-one-area-it-wont-completely-change-greycrofts-dana-settle-182051256.html)


[medical_debt_positive](https://finance.yahoo.com/news/millions-poised-to-get-a-better-credit-score-after-medical-debt-dropped-from-reports-210927590.html)


[sp_positive](https://finance.yahoo.com/news/stifel-raises-sp-forecast-citing-economic-resilience-164216721.html)


[stock_market_negative](https://finance.yahoo.com/news/stocks-slump-as-regional-banks-tank-stock-market-news-today-153658149.html)


[paramaount_negative](https://finance.yahoo.com/news/paramount-earnings-first-quarter-2023-may-4-112140604.html)


[apple_neutral](https://finance.yahoo.com/news/apple-isnt-playing-the-ai-hype-game-190726240.html)


[negative_interest_rate](https://finance.yahoo.com/news/federal-reserve-interest-rate-decision-may-3-155524134.html)


[negative_warren_buffet](https://finance.yahoo.com/news/buffett-on-the-regional-bank-crisis-messed-up-incentives-and-poor-communication-211138275.html)


[carvana_positive](https://finance.yahoo.com/news/carvana-stock-surges-as-used-car-dealer-sees-q2-profit-135040050.html)


[banks_neg](https://finance.yahoo.com/news/close-190-banks-could-face-163717073.html)


[unforadableBank_negative](https://finance.yahoo.com/news/housing-unaffordable-banks-losing-money-014524600.html)


[stockDividend_postive](https://finance.yahoo.com/news/6-7-yielding-dividend-etf-213300385.html)


[Mortgage_negative](https://finance.yahoo.com/news/housing-market-2023-prices-now-171702377.html)

[portfolio_neutral](https://finance.yahoo.com/video/portfolio-diversification-really-important-millennial-194044581.html)


[investing_neutral](https://www.yahoo.com/news/invest-stocks-beginner-guide-100009203.html)


[lifestyle_neutral](https://www.yahoo.com/lifestyle/there-are-two-types-of-stocks-on-robinhood-181831860.html)


[economy_postive](https://finance.yahoo.com/news/us-economy-has-regained-growth-momentum-in-april-as-recession-fears-swirl-161520510.html)


[tech_positive](https://finance.yahoo.com/news/why-tech-stocks-doing-well-150903541.html)


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
    header [''
    \begin{tabular}{|l|c|c|c|}
    \hline
    \textbf{Model and Training Regime} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} \\ \hline
    '''
    rows = []
    for label, res in zip(labels, results):
        row = f"{label} & {res[0]:.4f} & {res[1]:.4f} & {res[2]:.4f} \\\\ \\hline"
        rows.append(row)

    footer [''\end{tabular}'''

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

We also created a custom ModelEvaluator class, this class used a method analyze_article() which pulled the text from given Yahoo Finance articles, tokenized them, and made predictions on this unlabeled data. These predictions were used as a metric compared against our own opinions about which class each article fell into , in order to judge performance. 

```
def get_article_text(url):
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

  response = requests.get(url, headers=headers)
  if response.status_code == 200:
      soup = BeautifulSoup(response.content, "html.parser")

      # Find the article body
      article_body = soup.find("div", class_="caas-body")

      # Extract all the paragraph texts
      paragraphs = article_body.find_all("p")
      text = " ".join([p.text for p in paragraphs])

  else:
      print(f"Failed to download the webpage. Status code: {response.status_code}")
      text = ""

  return [text]
```

```
class ModelEvaluator():

  def __init__(self, model_path, model_type='distilbert'):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model.to(self.device)
    self.model_type = model_type

  def analyze_article(self, url):
    # Get the article text
    data = get_article_text(url)

    # Tokenize the text
    unlabeled_dataset = DatasetLoaderSingle(data, self.tokenizer, labels=None)
    self.model.eval()
    predictions = []
    label_dict = {2:'positive', 0:'negative', 1:'neutral'}

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(unlabeled_dataset):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if self.model_type == 'electra':
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            label = label_dict[preds[0]]
            predictions.append(label)

    return predictions

  def calculate_metrics(self, predictions, labels):
    y_pred = predictions
    y_true = labels
    label_dict = {'positive': 2, 'negative': 0, 'neutral':1}
    y_true = [label_dict[val] for val in y_true]
    y_pred = [label_dict[val] for val in y_pred]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

```