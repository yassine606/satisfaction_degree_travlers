import pickle

import numpy as np

from flask import render_template, redirect, url_for,request
from flask import Flask

from model import logreg

app = Flask(__name__)

RF=pickle.load(open('/Users/admin/PycharmProjects/app/model.pkl','rb'))

@app.route("/", methods=['GET','POST'])
def home():
 return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['Gender'])
        glucose = int(request.form['customer_type'])
        bp = int(request.form['age'])
        st = int(request.form['type_of_travel'])
        insulin = int(request.form['customer_class'])
        bmi = float(request.form['flight_distance'])
        dpf1 = float(request.form['inflight_wifi_service'])
        dpf2= float(request.form['departure_arrival_time_convenient'])
        dpf3 = float(request.form['ease_of_online_booking'])
        dpf4 = float(request.form['gate_location'])
        dpf5= float(request.form['food_and_drink'])
        dpf6= float(request.form['online_boarding'])
        dpf7 = float(request.form['seat_comfort'])
        dpf8 = float(request.form['inflight_entertainment'])
        dpf9 = float(request.form['onboard_service'])
        dpf11 = float(request.form['leg_room_service'])
        dpf12 = float(request.form['baggage_handling'])
        dpf13 = float(request.form['checkin_service'])
        dpf14 = float(request.form['inflight_service'])
        dpf15 = float(request.form['cleanliness'])
        dpf16 = float(request.form['departure_delay_in_minutes'])
        dpf17 = float(request.form['arrival_delay_in_minutes'])


        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf1,dpf2,dpf3,dpf4,dpf5,dpf6,dpf7,dpf8
                          ,dpf9,dpf11,dpf12,dpf13,dpf14,dpf15,dpf16,dpf17]])



        return render_template('result.html', prediction=logreg.predict(data))


if __name__== "__main__":
    app.run(host='localhost', port=5000)
