from flask import Flask, render_template_string, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ------------- إعداد البيانات والنموذج -------------
nltk.download('punkt')
nltk.download('stopwords')

# تحميل البيانات
df = pd.read_csv(r"C:\Users\Notebook\Desktop\final_balanced_data.csv")

# تنظيف الداتا
df = df[df['Column2'] != 'الدواء']
df = df[df['Column2'] != 'medicine']
df = df[df['Column2'] != 'Column2']
df = df.dropna(subset=['Column1', 'Column2'])
df = df.drop_duplicates(subset=['Column1', 'Column2'])

# إعداد أدوات اللغة العربية
stop_words = set(stopwords.words('arabic'))
stemmer = SnowballStemmer("arabic")

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    return text

def remove_noise(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_noise(text)
    text = normalize_arabic(text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

df['cleaned_col1'] = df['Column1'].apply(preprocess_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
X = vectorizer.fit_transform(df['cleaned_col1'])
y = df['Column2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ------------- دالة التوصية -------------
def recommend_medicine(symptoms):
    processed_symptoms = preprocess_text(symptoms)
    x_new = vectorizer.transform([processed_symptoms])
    predicted_medicine = clf.predict(x_new)[0]
    info = df[df['Column2'] == predicted_medicine]
    if not info.empty:
        price = info.iloc[0]['Column4'] if 'Column4' in df.columns else "غير متوفر"
        side_effects = info.iloc[0]['Column5'] if 'Column5' in df.columns else "غير متوفر"
    else:
        price, side_effects = "غير متوفر", "غير متوفر"
    return predicted_medicine, price, side_effects

# ------------- Flask App -------------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="UTF-8">
<title>نظام توصية أدوية</title>
<style>
body { font-family: Arial, sans-serif; background: #f2f2f2; padding: 30px; }
.container { background: #fff; padding: 20px; max-width: 600px; margin: auto; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);}
h2 { text-align: center; color: #333; }
input[type=text] { width: 100%; padding: 10px; margin: 10px 0; border-radius:5px; border:1px solid #ccc;}
input[type=submit] { background:#4CAF50; color:white; padding:10px; border:none; border-radius:5px; cursor:pointer;}
input[type=submit]:hover { background:#45a049;}
.result { background:#e7f3fe; padding:10px; border-left:5px solid #2196F3; margin-top:10px;}
</style>
</head>
<body>
<div class="container">
<h2>نظام توصية أدوية</h2>
<form method="post">
<input type="text" name="symptoms" placeholder="اكتب الأعراض هنا" required>
<input type="submit" value="توصية">
</form>
{% if result %}
<div class="result">
<p><b>الدواء المقترح:</b> {{ result[0] }}</p>
<p><b>السعر:</b> {{ result[1] }}</p>
<p><b>الآثار الجانبية:</b> {{ result[2] }}</p>
</div>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        symptoms = request.form["symptoms"]
        result = recommend_medicine(symptoms)
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(debug=True)
