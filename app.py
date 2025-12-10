import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import preprocessor
import helper
import matplotlib as mpl
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re

st.set_page_config(layout="wide")
st.title("CONNECTIFY 📊")

# Upload section
uploaded_file = st.file_uploader("Choose a chat file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.sidebar.title("Chat Info")
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Select user for analysis", user_list)

    if st.sidebar.button("Show Analysis"):

        # Basic Stats
        st.header("Top Statistics")
        num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)

        # Metric + Expanders for Interactive Details
        with col1:
            st.metric("Total Messages", num_messages)
            with st.expander("Click to view messages"):
                if selected_user == "Overall":
                    st.dataframe(df[['date', 'user', 'message']])
                else:
                    st.dataframe(df[df['user'] == selected_user][['date', 'user', 'message']])

        with col2:
            st.metric("Total Words", num_words)
            with st.expander("Click to view words"):
                word_df = df.copy()
                if selected_user != "Overall":
                    word_df = word_df[word_df['user'] == selected_user]
                word_df['tokens'] = word_df['message'].apply(lambda msg: word_tokenize(str(msg).lower()))
                word_df = word_df.explode('tokens').reset_index(drop=True)
                st.dataframe(word_df[['date', 'user', 'tokens']])

        with col3:
            st.metric("Media Shared", num_media)

        with col4:
            st.metric("Links Shared", num_links)
            with st.expander("Click to view links"):
                if selected_user == "Overall":
                    # Filtering the dataframe to show messages containing links
                    links_df = df[df['message'].str.contains("http|www", na=False)][['date', 'user', 'message']]
                else:
                    # Filtering the dataframe for a selected user
                    links_df = df[(df['user'] == selected_user) & (df['message'].str.contains("http|www", na=False))][
                        ['date', 'user', 'message']]
                # Extract links from the message using regular expressions
                def extract_link(message):
                    # Regex pattern to find URLs starting with http or www
                    urls = re.findall(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
                    if urls:
                        # Return the first URL found
                        return f'<a href="{urls[0]}" target="_blank">{urls[0]}</a>'
                    return ''
                # Apply the link extraction function to the dataframe
                links_df['link'] = links_df['message'].apply(extract_link)
                # Display the dataframe with clickable links (showing only the link column)
                for _, row in links_df.iterrows():
                    if row['link']:
                        st.markdown(f"**{row['user']}** on {row['date']}: {row['link']}")

        # Monthly Timeline
        st.header("Monthly Activity")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        col1, col2, col3 = st.columns([1, 3, 1])  # center col2 is wider
        with col2:
            st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        col1, col2, col3 = st.columns([1, 3, 1])  # center col2 is wider
        with col2:
            st.pyplot(fig)

        # Activity Map
        st.header("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='skyblue')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Heatmap
        st.header("Weekly Activity Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap, cmap="YlGnBu", ax=ax)
        col1, col2, col3 = st.columns([1, 4, 1])  # center col2 is wider
        with col2:
            st.pyplot(fig)

        # Most Active Users
        if selected_user == 'Overall':
            st.header("Most Active Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation="vertical")
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Word Cloud
        st.header("Most Common Words (WordCloud)")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        col1, col2, col3 = st.columns([1, 4, 1])  # center col2 is wider
        with col2:
            st.pyplot(fig)

        # Most Common Words (Bar Chart)
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        col1, col2, col3 = st.columns([1, 4, 1])  # center col2 is wider
        with col2:
            st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        mpl.rcParams['font.family'] = 'Segoe UI Emoji'
        with col2:
            if not emoji_df.empty:
                if list(emoji_df.columns) == [0, 1]:
                    emoji_df.columns = ['emoji', 'count']
                fig, ax = plt.subplots()
                ax.pie(
                    emoji_df['count'].head(),
                    labels=emoji_df['emoji'].head(),
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 16}
                )
                ax.set_title("Top Emojis", fontsize=18)
                st.pyplot(fig)
            else:
                st.write(f"No emojis are sent by {selected_user}.")

        # Sentiment Analysis
        st.header("Sentiment Analysis ")
        analyzer = SentimentIntensityAnalyzer()
        if selected_user != 'Overall':
            df_sent = df[df['user'] == selected_user].copy()
        else:
            df_sent = df.copy()
        df_sent['sentiment'] = df_sent['message'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df_sent['sentiment_label'] = df_sent['sentiment'].apply(lambda x: 'Positive 😊' if x >= 0.05 else ('Negative 😠' if x <= -0.05 else 'Neutral 😐'))
        sentiment_summary = df_sent['sentiment_label'].value_counts(normalize=True).reset_index()
        sentiment_summary.columns = ['Sentiment', 'Proportion']
        st.dataframe(sentiment_summary)

        fig, ax = plt.subplots()
        ax.pie(sentiment_summary['Proportion'], labels=sentiment_summary['Sentiment'], autopct='%1.1f%%',
               colors=['lightgreen', 'lightcoral', 'lightgray'])
        ax.set_title(f"Sentiment Distribution - {selected_user}")
        col1, col2, col3 = st.columns([1, 4, 1])  # center col2 is wider
        with col2:
            st.pyplot(fig)

        # Conversation Starter Detection
        if selected_user == 'Overall':
            st.header("Conversation Starters")
            df_sorted = df.sort_values(by='date')
            df_sorted['prev_date'] = df_sorted['date'].shift(1)
            df_sorted['gap_minutes'] = (df_sorted['date'] - df_sorted['prev_date']).dt.total_seconds() / 60
            df_sorted['new_convo'] = df_sorted['gap_minutes'] > 60
            starter_df = df_sorted[df_sorted['new_convo'] == True]
            top_starters = starter_df['user'].value_counts().reset_index()
            top_starters.columns = ['User', 'Conversations Started']
            st.dataframe(top_starters)
            fig, ax = plt.subplots()
            ax.bar(top_starters['User'], top_starters['Conversations Started'], color='salmon')
            plt.xticks(rotation='vertical')
            col1, col2, col3 = st.columns([1, 4, 1])  # center col2 is wider
            with col2:
                st.pyplot(fig)

            # Conversation Flow Graph
            st.header("Conversation Flow Graph (Interactive)")
            convo_edges = []
            last_user = None
            for index, row in df_sorted.iterrows():
                if last_user and row['user'] != last_user:
                    convo_edges.append((last_user, row['user']))
                last_user = row['user']

            edge_df = pd.DataFrame(convo_edges, columns=['from', 'to'])
            edge_counts = edge_df.value_counts().reset_index(name='count')
            filtered_edges = edge_counts[edge_counts['count'] > 2]

            G = nx.DiGraph()
            for _, row in filtered_edges.iterrows():
                G.add_edge(row['from'], row['to'], weight=row['count'])

            pos = nx.spring_layout(G, k=1, seed=42)
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node}<br>Messages: {G.degree[node]}")

            edge_x = []
            edge_y = []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='gray'),
                hoverinfo='none', mode='lines'))

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[node for node in G.nodes()],
                textposition="bottom center",
                marker=dict(
                    size=[10 + 3 * G.degree(n) for n in G.nodes()],
                    color='lightblue',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertext=node_text
            ))

            fig.update_layout(
                title="User Interaction Network (Filtered & Interactive)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)
