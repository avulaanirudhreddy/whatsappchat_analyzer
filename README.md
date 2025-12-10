# whatsappchat_analyzer
# CONNECTIFY – WhatsApp Chat Analyzer

A data-driven WhatsApp Chat Analyzer built using **Python** and **Streamlit**, designed to extract, process, and visualize WhatsApp conversations with meaningful insights.

---

## 🚀 Live Demo
https://whatsappchatanalyzer-l9pvn6j88fwsmjtbb2bj7h.streamlit.app/

---

## 📚 Overview

CONNECTIFY transforms raw WhatsApp chat exports into an interactive dashboard. The system performs:

* Data cleaning and formatting
* User-wise and overall statistics
* Time-based activity insights
* Word and emoji analytics
* Sentiment detection
* Conversation flow mapping

All outputs are displayed through an intuitive and interactive UI.

---

## 🎯 Key Features

### 1. Basic Statistics

* Total messages
* Word count
* Media shared
* Hyperlinks detected

### 2. Timelines & Activity Mapping

* Monthly timeline
* Daily timeline
* Most active days
* Most active months
* Hour-wise heatmap

### 3. Word & Emoji Analysis

* WordCloud generation
* Top 20 common words
* Emoji frequency charts

### 4. Sentiment Analysis

* Positive, negative, and neutral classification
* VADER-based scoring
* Pie-chart distribution

### 5. Conversation Flow & Starters

* Detects conversation initiators
* Finds long gaps between messages
* Creates a user-to-user interaction graph

---

## 🏗️ System Architecture

### Preprocessor Module (preprocessor.py)

* Extracts datetime, sender, and message
* Handles system notifications
* Produces structured DataFrame
* Adds day, month, hour, period, etc.

### Helper Module (helper.py)

* Computes message statistics
* Builds timelines
* Generates wordclouds
* Performs emoji, sentiment, and activity analysis

### UI Module (app.py)

* Built using Streamlit
* Handles file uploads
* Displays all charts & tables
* Renders insights across sections

---

## 📁 Project Structure

```
📦 whatsappchat_analyzer
 ┣ app.py
 ┣ helper.py
 ┣ preprocessor.py
 ┣ stop_hinglish.txt
 ┣ requirements.txt
 ┣ runtime.txt (optional)
 ┗ README.md
```

---

## 🛠 Installation & Running Locally

### 1. Clone Repository

```
git clone https://github.com/your-username/whatsappchat_analyzer.git
cd whatsappchat_analyzer
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

### 3. Run Application

```
streamlit run app.py
```

---

## 📝 How to Use

1. Export your WhatsApp chat as `.txt`
2. Upload the file into CONNECTIFY
3. Select a user or "Overall"
4. Click **Show Analysis**
5. Explore timelines, words, emojis, and sentiment

---

## 📦 Technologies Used

* Python
* Streamlit
* Pandas
* Matplotlib
* Seaborn
* NetworkX
* Plotly
* NLTK
* WordCloud
* URLEXTRACT
* Emoji
* VADER Sentiment

---

## 🔒 Limitations

* Works best with English/Hinglish chats
* Sentiment accuracy varies for sarcasm
* Large chat exports may load slower

---

## 📌 Future Enhancements

* Multi-language support
* Media analysis (images, audio)
* Topic modeling
* Chat similarity scoring
