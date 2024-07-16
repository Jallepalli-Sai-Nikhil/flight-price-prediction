from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

app = Flask(__name__)

# Load the pickled model
model_file = open('flight_price_prediction_model.pkl', 'rb')
model = pickle.load(model_file)
model_file.close()

# Load dataset and drop 'Unnamed: 0'
df = pd.read_csv('dataset.csv')
df.drop("Unnamed: 0", axis=1, inplace=True)

# Define categorical columns for ordinal encoding
ordinal_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

# Calculate unique values for each column (for dropdowns)
unique_values = {}
for col in df.columns:
    if col != 'price' and col != 'flight_encoded':  # Exclude 'price' and 'flight_encoded'
        unique_values[col] = df[col].unique().tolist()

# Encode 'flight' column using LabelEncoder if present
if 'flight' in df.columns:
    label_encoder = LabelEncoder()
    df['flight_encoded'] = label_encoder.fit_transform(df['flight'])
    df.drop('flight', axis=1, inplace=True)  # Drop original 'flight' column after encoding

# Home page with form for prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Extract form data
        data = {}
        for col in df.columns:
            if col != 'price' and col != 'flight_encoded':  # Exclude 'price' and 'flight_encoded'
                if col == 'flight':
                    flight_text = request.form['flight']
                    if flight_text in label_encoder.classes_:
                        data['flight_encoded'] = label_encoder.transform([flight_text])[0]
                    else:
                        # Handle new flight number not seen during training
                        data['flight_encoded'] = label_encoder.transform(['unknown'])[0]
                elif col == 'duration':
                    data['duration'] = [request.form['duration']]  # Include 'duration' from form
                elif col == 'days_left':
                    data['days_left'] = [request.form['days_left']]  # Include 'days_left' from form
                else:
                    data[col] = [request.form[col]]

        # Convert to DataFrame
        custom_df = pd.DataFrame(data)

        # Perform ordinal encoding for categorical features
        ordinal_encoder = OrdinalEncoder()
        custom_df[ordinal_cols] = ordinal_encoder.fit_transform(custom_df[ordinal_cols])

        # Ensure all required columns are present and in the correct order
        for col in df.columns:
            if col not in custom_df.columns:
                custom_df[col] = 0  # Add missing columns if any, with default values

        # Handle 'duration' and 'days_left' columns specifically (assuming they're not in form)
        if 'duration' not in custom_df.columns:
            custom_df['duration'] = 0  # Or handle as needed based on your model requirements

        if 'days_left' not in custom_df.columns:
            custom_df['days_left'] = 0  # Or handle as needed based on your model requirements

        custom_df = custom_df[df.columns.drop('price')]  # Reorder columns to match model's expected order, excluding 'price'

        # Make prediction
        prediction = model.predict(custom_df)

        return render_template('index.html', df=df.columns, ordinal_cols=ordinal_cols, unique_values=unique_values, prediction=f'Predicted Price: ₹{prediction[0]:.2f}')
    
    # Render the form with input fields for all columns except 'price' and 'flight_encoded'
    return render_template('index.html', df=df.columns, ordinal_cols=ordinal_cols, unique_values=unique_values, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
