from flask import Flask, request, render_template_string, redirect, url_for
import pandas as pd
import joblib
from datetime import datetime
import ipaddress

app = Flask(__name__)

# Load models and data
rf = joblib.load(r"C:\Users\cskbu\OneDrive\Desktop\fraudde\rf_model.pkl")
le = joblib.load(r"C:\Users\cskbu\OneDrive\Desktop\fraudde\label_encoder.pkl")
ip_data = pd.read_csv(r"C:\Users\cskbu\OneDrive\Desktop\fraudde\IpAddress_to_Country.csv")

uploaded_df = None

# ------------------- Stylish HTML TEMPLATE ------------------- #
html_template = """
<html>
<head>
    <title>Fraud Detection Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 40px;
            background: linear-gradient(to right, #1c1c1c, #2c2c2c);
            font-family: 'Poppins', sans-serif;
            color: #f0f0f0;
        }
        h2 {
            font-size: 36px;
            color: #ff3c3c;
            margin-bottom: 20px;
        }
        h3 {
            font-size: 24px;
            margin-top: 40px;
            color: #ffa07a;
        }
        form {
            margin-bottom: 20px;
        }
        input[type="file"],
        input[type="submit"],
        button {
            background-color: #ff3c3c;
            color: white;
            padding: 12px 20px;
            margin-right: 10px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.3s ease;
        }
        input[type="file"] {
            background-color: #444;
            border: 1px solid #777;
        }
        input[type="submit"]:hover,
        button:hover {
            background-color: #e63946;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 10px;
            margin-top: 20px;
        }
        th, td {
            padding: 12px 15px;
            background-color: #333;
            border-bottom: 1px solid #444;
        }
        th {
            background-color: #444;
            font-weight: 600;
            color: #ffbaba;
        }
        tr:nth-child(even) td {
            background-color: #2a2a2a;
        }
        ul {
            padding-left: 20px;
        }
        ul li {
            margin-bottom: 5px;
        }
        a button {
            text-decoration: none;
        }
        @media (max-width: 768px) {
            body {
                padding: 20px;
            }
            table, thead, tbody, th, td, tr {
                display: block;
            }
            tr {
                margin-bottom: 15px;
            }
            td {
                padding-left: 50%;
                position: relative;
            }
            td::before {
                position: absolute;
                left: 15px;
                top: 12px;
                white-space: nowrap;
                font-weight: bold;
            }
        }
    </style>
</head>
<body>
    <h2>🛡️ E-commerce Fraud Detection</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
        {% if data_uploaded %}
            <a href="#fraud-section"><button formaction="/detect">Detect Fraud</button></a>
        {% endif %}
    </form>

    {% if data %}
    <h3>📄 Uploaded Transactions</h3>
    <table>
        <tr>
            <th>S.No</th>
            {% for col in data[0].keys() %}
                <th>{{ col }}</th>
            {% endfor %}
        </tr>
        {% for row in data %}
            <tr>
                <td>{{ loop.index }}</td>
                {% for val in row.values() %}
                    <td>{{ val }}</td>
                {% endfor %}
            </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if fraud_results %}
    <div id="fraud-section">
        <h3>🚨 Fraud Predictions</h3>
        <table>
            <tr>
                <th>S.No</th>
                <th>User ID</th>
                <th>Reasons</th>
            </tr>
            {% for item in fraud_results %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ item.user_id }}</td>
                    <td>
                        <ul>
                            {% for reason in item.reasoning %}
                                <li>{{ reason|safe }}</li>
                            {% endfor %}
                        </ul>
                    </td>
                </tr>
            {% endfor %}
        </table>
    </div>
    <script>
        window.location.hash = "fraud-section";
    </script>
    {% endif %}
</body>
</html>
"""
# ------------------------------------------------------------ #

# Country lookup from IP range
def get_country(ip_int):
    row = ip_data[(ip_data['lower_bound_ip_address'] <= ip_int) & (ip_data['upper_bound_ip_address'] >= ip_int)]
    return row.iloc[0]['country'] if not row.empty else 'others'

# Label encode with fallback for unknowns
def safe_label_encode(values, encoder, unknown_label='others'):
    classes = encoder.classes_.tolist()
    values = [val if val in classes else unknown_label for val in values]
    return encoder.transform(values)

# Main preprocessing logic
def preprocess(df):
    df.columns = df.columns.str.strip()
    required_cols = {'user_id', 'signup_time', 'purchase_time', 'ip_address', 'device_id', 'source', 'browser', 'sex'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    datetimeFormat = '%d-%m-%Y %H:%M'
    df['diff_time'] = df.apply(lambda row: (
        (datetime.strptime(row['purchase_time'], datetimeFormat) - datetime.strptime(row['signup_time'], datetimeFormat)).total_seconds()
    ) if pd.notnull(row['signup_time']) and pd.notnull(row['purchase_time']) else 0, axis=1)

    df.drop(columns=['signup_time', 'purchase_time'], inplace=True)

    df['ip_int'] = df['ip_address'].apply(lambda ip: int(ipaddress.ip_address(ip)) if pd.notnull(ip) else 0)
    df['country'] = df['ip_int'].apply(get_country)

    df['num_used_device'] = df.groupby('device_id')['user_id'].transform('nunique')
    df['num_ip_repeat'] = df.groupby('ip_address')['user_id'].transform('nunique')

    for col in ['device_id', 'source', 'browser', 'sex', 'country']:
        df[col] = safe_label_encode(df[col].astype(str), le)

    X = df.drop(columns=['user_id'])
    X = X[rf.feature_names_in_]
    preds = rf.predict(X)

    df['is_fraud_predicted'] = preds
    fraud_df = df[df['is_fraud_predicted'] == 1]
    output = []

    others_encoded = le.transform(['others'])[0]

    for i, row in fraud_df.iterrows():
        features = X.loc[i]
        reasoning = []

        # Extract enriched info
        product_name = row.get('product_name', 'Unknown Product')
        product_id = row.get('product_id', 'N/A')
        product_link = row.get('product_link', '')
        value = row.get('purchase_value', 'unknown')

        if features['diff_time'] < 600:
            reasoning.append(f"User purchased within {int(features['diff_time'])} seconds of signing up.")
        if features['num_used_device'] > 2:
            reasoning.append(f"Device reused by {int(features['num_used_device'])} users.")
        if features['num_ip_repeat'] > 5:
            reasoning.append(f"IP used by {int(features['num_ip_repeat'])} different users.")
        if features['country'] == others_encoded:
            reasoning.append("Transaction originated from an unknown/high-risk country.")

        # Product details
        product_details = f"Product: {product_name} (ID: {product_id})"
        if value != 'unknown':
            product_details += f" | Value: ₹{value}"
        if product_link:
            product_details += f" | <a href='{product_link}' target='_blank'>Amazon Link</a>"

        reasoning.insert(0, product_details)

        output.append({
            "user_id": row['user_id'],
            "reasoning": reasoning or ["Model flagged this transaction based on complex patterns."]
        })

    return output

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_df
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template_string(html_template, data_uploaded=False)
        try:
            uploaded_df = pd.read_csv(file)
            return render_template_string(html_template, data_uploaded=True, data=uploaded_df.to_dict(orient='records'))
        except Exception as e:
            return render_template_string(html_template, data_uploaded=False, fraud_results=[{"user_id": "Error", "reasoning": [str(e)]}])
    return render_template_string(html_template, data_uploaded=False)

@app.route("/detect", methods=["POST"])
def detect():
    global uploaded_df
    if uploaded_df is None:
        return redirect(url_for('index'))
    try:
        fraud_results = preprocess(uploaded_df)
        return render_template_string(html_template, data_uploaded=True,
                                      data=uploaded_df.to_dict(orient='records'),
                                      fraud_results=fraud_results)
    except Exception as e:
        return render_template_string(html_template, data_uploaded=True,
                                      data=uploaded_df.to_dict(orient='records'),
                                      fraud_results=[{"user_id": "Error", "reasoning": [str(e)]}])

if __name__ == "__main__":
    app.run(debug=True)
