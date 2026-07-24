from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar el pipeline que guardamos previamente
with open('pipeline.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener el JSON enviado
    data = request.get_json()
    
    # Convertirlo directamente en un DataFrame de Pandas
    input_data = pd.DataFrame([data])
    
    # Hacer la predicción (El pipeline preprocesará automáticamente los datos)
    prediccion = modelo.predict(input_data)
    
    # Retornar respuesta
    return jsonify({'Survived': int(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True)