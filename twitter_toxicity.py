import twint
from transformers import pipeline
import numpy as np

tweets = []
c = twint.Config()
c.Username = "lexfridman"
c.Limit = 1000
c.Store_object = True
c.Hide_output = True
c.Store_object_tweets_list = tweets
twint.run.Search(c)

print(f"Tweets fetched: {len(tweets)}")
classifier = pipeline('sentiment-analysis')
results = classifier([t.tweet for t in tweets])

positive = [r for r in results if r['label'] == 'POSITIVE']
negative = [r for r in results if r['label'] == 'NEGATIVE']
print(f"Positive: {len(positive) / float(len(results)) * 100}% "
      f"with average score {np.round(np.sum([r['score'] for r in positive]) / float(len(positive)), 4)}")
print(f"Negative: {len(negative) / float(len(results)) * 100}% "
      f"with average score {np.round(np.sum([r['score'] for r in negative]) / float(len(negative)), 4)}")

# Output:
# Tweets fetched: 1000
# Positive: 62.4% with average score 0.9574
# Negative: 37.6% with average score 0.9298
