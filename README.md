# Sentiment Analysis with BERT & Web Scraping

## Description
This project is a sentiment analysis tool that uses a pre-trained BERT model to analyze reviews scraped from the web. The model predicts the sentiment of the reviews on a scale of 1 to 5. The data is collected from websites using web scraping techniques with `requests` and `BeautifulSoup`.

## Features
- **Sentiment Analysis**: Uses `nlptown/bert-base-multilingual-uncased-sentiment` for analyzing text sentiment.
- **Web Scraping**: Extracts reviews from Yelp pages.
- **Deep Learning**: Utilizes PyTorch and Hugging Face's `transformers` library.

## Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers requests beautifulsoup4 pandas numpy
```

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Function to analyze sentiment
def analyze_sentiment(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)
    sentiment = int(torch.argmax(result.logits)) + 1
    return sentiment

# Web scraping Yelp reviews
url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
reviews = [tag.text for tag in soup.find_all('p', {'class': regex})]

# Analyze sentiment for each review
sentiments = [analyze_sentiment(review) for review in reviews]
print(sentiments)
```

## Contributing
Feel free to contribute by opening an issue or submitting a pull request.

## License
This project is licensed under the MIT License.

