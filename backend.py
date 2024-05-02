from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def front_page():
    return render_template('liver.html')

@app.route('/liver', methods=['GET', 'POST'])
def liver_prediction():
    if request.method == 'POST':
        age=request.form['Age']
        total_bilirubin = request.form['Total Bilirubin']
        direct_bilirubin = request.form['Direct_Bilirubin']
        alkaline_phosphotase = request.form['Alkaline_Phosphotase']
        alamine_aminotransferase = request.form['Alamine_Aminotransferase']
        aspartate_aminotransferase = request.form['Aspartate_Aminotransferase']
        total_proteins = request.form['Total_Protiens']
        albumin = request.form['Albumin']
        albumin_and_globulin_ratio = request.form['Albumin_and_Globulin_Ratio']
        
        model = pickle.load(open(r'C:/Users/Dell/OneDrive/Desktop/Predict Liver Disease/model/liver_model.pkl', 'rb'))
        
        input_data = (total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        prediction = model.liver_prediction(input_data_reshaped)
        
        if prediction[0] == 2:
            senddata = 'According to the given details, the person does not have Liver Disease.'
        else:
            senddata = 'According to the given details, chances of having Liver Disease are high. Please consult a doctor.'
        
        return render_template('result.html', resultvalue=senddata)

if __name__ == "__main__":
    app.run(debug=True)
