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