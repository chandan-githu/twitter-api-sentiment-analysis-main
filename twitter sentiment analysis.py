import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def preprocess_tweet(tweet):
    """Preprocess the tweet by replacing usernames and URLs."""
    tweet_words = []
    for word in tweet.split():
        if word.startswith('@') and len(word) > 1:
            word = '@user'  # Replace usernames
        elif word.startswith('http'):
            word = 'http'   # Replace URLs
        tweet_words.append(word)
    return " ".join(tweet_words)

def analyze_sentiment(tweet):
    """Analyze sentiment of the given tweet and visualize results."""
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess tweet
    tweet_proc = preprocess_tweet(tweet)

    # Sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    # Extract scores
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    # Labels for sentiment
    labels = ['Negative', 'Neutral', 'Positive']

    # Print results
    for label, score in zip(labels, scores):
        print(f"{label}: {score:.4f}")

    # Set the style
    sns.set(style="darkgrid")  # Change the style here (e.g., "darkgrid", "whitegrid", "ticks", etc.)

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bar_colors = sns.color_palette("Set2")  # Using a different color palette
    plt.bar(labels, scores, color=bar_colors, edgecolor='black')
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Titles and labels
    plt.title('Sentiment Analysis Results', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)

    # Annotate scores on bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12, fontweight='bold')

    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tweet = 'you are very beautiful'
    analyze_sentiment(tweet)
