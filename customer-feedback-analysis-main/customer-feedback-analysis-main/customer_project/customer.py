import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/siddh/OneDrive/Desktop/customer-feedback-analysis-main/customer-feedback-analysis-main/customer_project/sentiment_analysis_feedback_3000_updated.csv")

# Print columns to verify column presence
print("Columns in dataset:", df.columns)

# Function to calculate sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to feedback texts
df['Sentiment'] = df['Customer Feedback Text'].apply(get_sentiment)

# 1. Sentiment Distribution Pie Chart
sentiment_counts = df['Sentiment'].value_counts()
print("Sentiment counts:", sentiment_counts)

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'gray'], startangle=140)
plt.title('Sentiment Analysis of Customer Feedback')
plt.ylabel('')  # Remove y-axis label for clarity
plt.show()
print("Sentiment pie chart completed")

# 2. Age Group Distribution Pie Chart
if 'Age of the Customer' in df.columns:
    df['Age of the Customer'] = pd.to_numeric(df['Age of the Customer'], errors='coerce')
    df = df.dropna(subset=['Age of the Customer'])
    age_bins = [0, 20, 40, 60, 80, 100]
    age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['Age Group'] = pd.cut(df['Age of the Customer'], bins=age_bins, labels=age_labels)
    age_group_counts = df['Age Group'].value_counts().sort_index()
    print("Age group counts:", age_group_counts)

    plt.figure(figsize=(8, 6))
    age_group_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Distribution of Age Groups')
    plt.ylabel('')
    plt.show()
    print("Age group pie chart completed")
else:
    print("The 'Age of the Customer' column is missing from the dataset.")

# 3. Gender Distribution Pie Chart
if 'Customer Gender' in df.columns:
    gender_counts = df['Customer Gender'].value_counts()
    print("Gender counts:", gender_counts)

    plt.figure(figsize=(8, 6))
    gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'pink'], startangle=140)
    plt.title('Distribution by Gender')
    plt.ylabel('')
    plt.show()
    print("Gender pie chart completed")
else:
    print("The 'Customer Gender' column is missing from the dataset.")

# 4. Platform Distribution Pie Char
if 'Platform Used' in df.columns:
    platform_counts = df['Platform Used'].value_counts()
    print("Platform counts:", platform_counts)

    plt.figure(figsize=(10, 6))
    platform_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Distribution by Platform')
    plt.ylabel('')
    plt.show()
    print("Platform pie chart completed")
else:
    print("The 'Platform Used' column is missing from the dataset.")

# Save the full dataset with sentiment and age group as a new CSV
df.to_csv("C:/Users/siddh/OneDrive/Desktop/customer-feedback-analysis-main/customer-feedback-analysis-main/customer_project/sentiment_feedback.csv", index=False)
