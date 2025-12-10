from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import string
import emoji
import nltk
nltk.download('punkt',force=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
extract=URLExtract()
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = df['message'].apply(lambda x: len(x.split())).sum()
    num_media_messages = df[df['message'].str.strip().str.startswith('<Media omitted>')].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages, words, num_media_messages,len(links)
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df=round(df['user'].value_counts() / df.shape[0] * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df
def create_wordcloud(selected_user, df):
    f = open("C:/Users/bahul/Desktop/IOMP/stop_hinglish.txt", 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted>', case=False, na=False)]
    temp = temp[~temp['message'].str.contains(r'\(file attached\)', case=False, na=False)]
    temp = temp[~temp['message'].str.contains('This message was deleted', case=False, na=False)]
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            word = word.strip(string.punctuation)
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc=wc.generate(df['message'].str.cat(sep=" "))
    return df_wc
def most_common_words(selected_user, df):
    f = open("C:/Users/bahul/Desktop/IOMP/stop_hinglish.txt", 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted', case=False, na=False)]
    temp = temp[~temp['message'].str.contains(r'\(file attached\)', case=False, na=False)]
    temp = temp[~temp['message'].str.contains('This message was deleted', case=False, na=False)]
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            word = word.strip(string.punctuation)
            if word and word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df
def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df
def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline
def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline
def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()
def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()
def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap
def compute_sentiment(df, selected_user):
    sia = SentimentIntensityAnalyzer()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df.copy()
    df['sentiment_score'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    def label_sentiment(score):
        if score >= 0.05:
            return 'Positive 😊'
        elif score <= -0.05:
            return 'Negative 😠'
        else:
            return 'Neutral 😐'
    df['sentiment'] = df['sentiment_score'].apply(label_sentiment)
    sentiment_summary = df['sentiment'].value_counts(normalize=True).reset_index()
    sentiment_summary.columns = ['Sentiment', 'Percentage']
    sentiment_summary['Percentage'] = sentiment_summary['Percentage'] * 100
    return df, sentiment_summary