import joblib
from flask import Flask, render_template, redirect, url_for,request,send_file
import random
import string
from PIL import Image, ImageDraw, ImageFont
import os

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sqlite3

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
#model1 = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
con=sqlite3.connect('database.db') 

# Set the upload folder path
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Set the download folder path
app.config['DOWNLOAD_FOLDER'] = 'static/downloads/'

# Set the secret key for encrypting session data
# app.secret_key = 'your-secret-key'


bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column('id',db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    
    # def __init__(self,username,email,password):
    #     self.username=username
    #     self.email=email 
    #     self.password=password 

# with app.app_context:
#     db.create_all()
@app.before_first_request
def create_tables():
    db.create_all()
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=34)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=34)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return render_template("login.html", form=form)
    return render_template("login.html", form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")
    return render_template('signup.html', form=form)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/disindex")

def disindex():
    return render_template("disindex.html")


@app.route("/cancer")
@login_required
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")


@app.route("/kidney")
@login_required
def kidney():
    return render_template("kidney.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
    if(int(result) == 1):
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return render_template("kidney_result.html", prediction_text=prediction)


@app.route("/liver")
@login_required
def liver():
    return render_template("liver.html")


def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Liver Disease"
    return render_template("liver_result.html", prediction_text=prediction)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

#commented code of breast cancer
# @app.route('/predict', methods=['POST'])
# def predict():
#     input_features = [int(x) for x in request.form.values()]
#     features_value = [np.array(input_features)]
#     features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
#                      'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
#     df = pd.DataFrame(features_value, columns=features_name)
#     output = model.predict(df)
#     if output == 4:
#         res_val = "a high risk of Breast Cancer"
#     else:
#         res_val = "a low risk of Breast Cancer"

#     return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))
###############################################################################################################33

# Function to generate a 6 digit unique ID
def generate_id():
    return ''.join(random.choices(string.digits, k=6))

# Route to handle form submission for health card
@app.route('/submit', methods=['POST'])
def submit_form():
    # Get form data
    name = request.form['name']
    dob = request.form['dob']
    address = request.form['address']
    gender = request.form['gender']
    contact = request.form['contact']
    bloodgroup = request.form['bloodgroup']
    profile_pic = request.files['profile_pic']

    # Generate a unique health ID
    health_id = generate_id()
   
    # Save the uploaded profile picture
    filename = profile_pic.filename
    filename=filename.split('.')
    filename=health_id+'.'+filename[1]
    profile_pic.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Create a new PIL Image object for the health card
    card_width = 700
    card_height = 500
    card_color = (250, 251, 255)
    card_image = Image.new('RGB', (card_width, card_height), color=card_color)

    # Draw the profile picture on the health card
    profile_pic_width = 200
    profile_pic_height = 200
    profile_pic_border = 5
    profile_pic_margin = 50
    profile_pic_image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).resize((profile_pic_width, profile_pic_height))
    card_image.paste(profile_pic_image, (profile_pic_margin, profile_pic_margin))
    draw = ImageDraw.Draw(card_image)
    draw.rectangle((profile_pic_margin - profile_pic_border, profile_pic_margin - profile_pic_border, 
                    profile_pic_margin + profile_pic_width + profile_pic_border, profile_pic_margin + profile_pic_height + profile_pic_border), 
                    outline=(0, 0, 0), width=profile_pic_border)
    
    # Draw the patient details on the health card
    text_color = (0, 0, 0)
    text_margin = 300
    text_font = ImageFont.truetype('arial.ttf', size=30)
    draw.text((text_margin, profile_pic_margin), f"Name: {name}", fill=text_color, font=text_font)
    draw.text((text_margin, profile_pic_margin+50), f"Date of Birth: {dob}", fill=text_color, font=text_font)
    draw.text((text_margin, profile_pic_margin+150), f"Gender: {gender}", fill=text_color, font=text_font)
    draw.text((text_margin, profile_pic_margin+200), f"Contact: {contact}", fill=text_color, font=text_font)
    draw.text((text_margin, profile_pic_margin+250), f"Address: {address}", fill=text_color, font=text_font)
    draw.text((text_margin, profile_pic_margin+300), f"Blood Group: {bloodgroup}", fill=text_color, font=text_font)
    draw.text((text_margin, profile_pic_margin+350), f"Health ID: {health_id}", fill=text_color, font=text_font)

# Save the health card image
    card_filename = f"{health_id}.png"
    card_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], card_filename)
    card_image.save(card_filepath)

# Render the form result template and pass the health ID and card URL
    return render_template('cancer_result.html', health_id=health_id,name=name,dob=dob,gender=gender,contact=contact,address=address,bloodgroup=bloodgroup,profile_pic_image=profile_pic_image,card_url=card_filename)

@app.route('/result', methods=['GET'])
def display_result():
# Get health ID from query parameters
    health_id = request.args.get('health_id')
    
    card_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], f"{health_id}.png")
    
    return send_file(card_filepath, as_attachment=True)
##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)


############################################################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
        bmi=0
        BMI=''
        if request.method=='POST' and 'weight' in request.form and 'height' in request.form:
            Weight =float(request.form.get('weight'))
            Height =float(request.form.get('height'))
            bmi=round(Weight/((Height/100)**2),2)
            if(bmi < 18.5):
                BMI = "BMI "+str(bmi)+" falls within the underweight range" 
            
            elif((bmi > 18.5) and (bmi<= 24.9)):
                
                BMI= "BMI "+str(bmi)+" falls within the normal or healthy weight range"

            elif((bmi >= 25) and (bmi <= 29.9 )):
                BMI = "BMI "+str(bmi)+" falls within the overweight range"
            
            else:
                BMI = "BMI "+str(bmi)+" falls within the obese range"
            
        
        return render_template('heart_result.html', prediction_text='Patients {}'.format(BMI))
    # input_features = [float(x) for x in request.form.values()]
    # features_value = [np.array(input_features)]

    # features_name = ["age", "trestbps", "chol", "thalach", "oldpeak", "sex_0",
    #                  "  sex_1", "cp_0", "cp_1", "cp_2", "cp_3", "  fbs_0",
    #                  "restecg_0", "restecg_1", "restecg_2", "exang_0", "exang_1",
    #                  "slope_0", "slope_1", "slope_2", "ca_0", "ca_1", "ca_2", "thal_1",
    #                  "thal_2", "thal_3"]

    # df = pd.DataFrame(features_value, columns=features_name)
    # output = model1.predict(df)

    # if output == 1:
    #     res_val = "a high risk of Heart Disease"
    # else:
    #     res_val = "a low risk of Heart Disease"

    # return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))


############################################################################################################

if __name__ == "__main__":
    #db.create_all()
    app.run(debug=True)

