# type "streamlit run fruit_vegetable.py" to run the .py file

import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Prediction"])

# Main Page
if(app_mode=="Home"):
    image_path = "D:\\Prodigy\\Food_Classification\\fruit-vegetable-classification-work\\fruit_vegetable.jpg"
    st.image(image_path)
    st.header("About Project")
    st.write("This is a Food recognisation system. You can insert any image of the fruits and vegetable listed below and get a quick response about the picture you inserted. Model can also predict the Calorie content in the food you will provide.")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")

    if test_image:
        st.subheader("The Image you selected is displayed below:")
        st.toast("Yummy!!! It's a Delicious Choice!!")
        st.image(test_image,width=4,use_column_width=True)
            
        #Predict button
        if st.button("Predict"):
            
            st.write("Our Prediction")
            model = load_model('model_r_which_fruit_vegetable.h5')
            Image = image.load_img(test_image, target_size = (64,64))
            Image = image.img_to_array(Image)
            Image=Image/255
            Image = np.expand_dims(Image, axis = 0)
            result = model.predict(Image)
            result_index = np.argmax(result)
            class_names = ['Apple', 'Banana', 'Beetroot', 'Bell Pepper', 'Cabbage', 'Capsicum', 'Carrot', 
                            'Cauliflower', 'Chilli Pepper', 'Corn', 'Cucumber', 'Eggplant', 'Garlic', 'Ginger', 
                            'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Lettuce', 'Mango', 'Onion', 'Orange', 'Paprika', 
                            'Pear', 'Peas', 'Pineapple', 'Pomegranate', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 
                            'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip', 'Watermelon']
            
            st.success("Model is Predicting that it's a {}".format(class_names[result_index]))

            st.snow()

            calories_dict = {"apple": 52,"banana": 89,"beetroot": 43,"bell pepper": 31,"cabbage": 25,"capsicum": 40,"carrot": 41,
                                "cauliflower": 25,"chilli pepper": 40,"corn": 96,"cucumber": 15,"eggplant": 25,"garlic": 149,"ginger": 80,
                                "grapes": 69,"jalapeño": 29,"kiwi": 61,"lemon": 29,"lettuce": 15,"mango": 60,"onion": 40,"orange": 47,
                                "paprika": 282,"pear": 57,"peas": 81,"pineapple": 50,"pomegranate": 83,"potato": 77,"raddish": 16,
                                "soy beans": 446,"spinach": 23,"sweetcorn": 86,"sweetpotato": 86,"tomato": 18,"turnip": 28,"watermelon": 30}
            calories = list(calories_dict.values())

            #st.success("Calorie content in the food is {}".format(calories[result_index]))
            st.warning(f"Calorie content in {class_names[result_index]} is {calories[result_index]}")

            Fruits = ["Banana", "Apple", "Pear", "Grapes", "Orange", "Kiwi", "Watermelon", "Pomegranate", "Pineapple", "Mango"]
            Vegetables = ["Cucumber", "Carrot", "Capsicum", "Onion", "Potato", "Lemon", "Tomato", "Radish", "Beetroot", "Cabbage", 
                        "Lettuce", "Spinach", "Soybean", "Cauliflower", "Bell Pepper", "Chilli Pepper", "Turnip", "Corn", 
                        "Sweetcorn", "Sweet Potato", "Paprika", "Jalepeno", "Ginger", "Garlic", "Peas", "Eggplant"]

            Predicted_food = class_names[result_index]
            if Predicted_food in Fruits:
                st.info("It belongs to the Fruit category")
            elif Predicted_food in Vegetables:
                st.info("It belongs to the Vegetable category")
            else:
                st.info("Can't predict")