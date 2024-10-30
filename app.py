from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import gdown

# Inicializa a aplicação Flask
app = Flask(__name__)

# Baixar o modelo do Google Drive
model_path = 'model_after_fine_tuning_1.keras'

if not os.path.exists(model_path):
 gdown.download('https://drive.google.com/uc?id=1hxDY7WN23Doi6LvDjVoTH3GISaDXG28C', 'model_after_fine_tuning_1.keras', quiet=False)


model = load_model(model_path)

# Define o diretório para uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']


# Rota principal
@app.route('/')
def index():
    return render_template('index.html')


# Rota para previsão
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Salva a imagem enviada
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Processa a imagem
    img = image.load_img(img_path, target_size=(480, 480))  # Ajuste o tamanho conforme necessário
    img_array = np.expand_dims(img, axis=0)

    # Faz a previsão
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)  # Obtém a classe prevista

    # Retorna a previsão
    return render_template('result.html', prediction=class_names[predicted_class[0]])


# Executa a aplicação
if __name__ == '__main__':
    app.run(debug=True, port=80)

# Dps no terminal tenho que colocar ngrok http 80, mas só dá quando o pc estiver a rodar o código