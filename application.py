import flask
from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("Model/phonescaler.pkl", "rb"))
model = pickle.load(open("Model/phonelogreg.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=" "

    if request.method=='POST':

        battery_power=int(request.form.get("battery_power"))
        bluetooth = float(request.form.get('blue',0.0))
        clock_speed = int(request.form.get('clock_speed'))
        dual_sim = int(request.form.get('dual_sim',0.0))
        Front_Camera_mpxl= int(request.form.get('fc',0.0))
        four_g = int(request.form.get('four_g'))
        int_memory = int(request.form.get('int_memory',0.0))
        Mobile_Depth_in_cm = float(request.form.get('m_dep',0.0))
        Weight_of_mobile_phone = int(request.form.get('mobile_wt',0.0))
        Number_of_cores_of_processor = int(request.form.get('n_cores',0.0))
        Primary_Camera_mega_pixels = int(request.form.get('pc',0.0))
        Pixel_Resolution_Height = int(request.form.get('px_height',0.0))
        Pixel_Resolution_Width = int(request.form.get('px_width',0.0))
        ram = int(request.form.get('ram',0.0))
        Screen_Height_of_mobile_in_cm = int(request.form.get('sc_h',0.0))
        Screen_Width_of_mobile_in_cm = int(request.form.get('sc_w',0.0))
        Maximum_Battery_Life_on_a_Single_Charge = int(request.form.get('talk_time',0.0))
        three_g = int(request.form.get('three_g',0.0))
        touch_screen = int(request.form.get('touch_screen',0.0))
        wifi = int(request.form.get('wifi',0.0))
        
      


        new_data=scaler.transform([[battery_power, bluetooth, clock_speed, dual_sim,Front_Camera_mpxl,four_g, int_memory,Mobile_Depth_in_cm,
                                    Weight_of_mobile_phone,Number_of_cores_of_processor, Primary_Camera_mega_pixels,Pixel_Resolution_Height
                                    , Pixel_Resolution_Width,ram,Screen_Height_of_mobile_in_cm,Screen_Width_of_mobile_in_cm, Maximum_Battery_Life_on_a_Single_Charge
                                     , three_g,touch_screen,wifi, ]])
        predict=model.predict(new_data)
       
        if predict[0] ==0 :
            result = 'Low Cost'
        elif predict[0]==1:
            result = 'Medium Cost'
        elif predict[0]==2:
            result = 'High Cost'
        else:
            result ='Very High Cost'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")