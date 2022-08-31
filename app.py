from matplotlib import image
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from os.path import dirname, join, realpath

# add banner image
st.header("Landslide prevention and management")
st.image("images/landslide1.png")
st.subheader(
    """
A simple app to predict whether a landslide occured or not.
"""
)

# form to collect user information
my_form = st.form(key="landslide_form")

elevation = my_form.number_input("Enter the elevation of that area", min_value=1, max_value=1000)

slope = my_form.number_input("Enter the slope of that area", min_value=0, max_value=80)

aspect = my_form.number_input("Enter the aspect of that area", min_value=18, max_value=360)

placurv = my_form.number_input("Enter the planform curvature of that area", min_value=-1, max_value=1)

procurv = my_form.number_input("Enter the profile curvature of that area", min_value=-1, max_value=1)

lsfactor = my_form.number_input("Enter the length slope factor of that area", min_value=0, max_value=60)

twi = my_form.number_input("Enter the topographic wetness index of that area", min_value=0, max_value=20)

sdoif = my_form.number_input("Enter the Step duration orographic intensification factor of that area", min_value=0, max_value=2)

geology = my_form.number_input("Enter the geology type of that area",min_value=1, max_value=7)

st.markdown("Where; 1: Weathered Cretaceous granitic rocks 2: Weathered Jurassic granite rocks 3: Weathered Jurassic tuff and lava 4: Weathered Cretaceous tuff and lava 5: Quaternary deposits 6: Fill 7: Weathered Jurassic sandstone, siltstone and mudstone")

submit = my_form.form_submit_button(label="Predict")

# load the model and one-hot-encoder and scaler

with open(
    join(dirname(realpath(__file__)), "model/catboost-model2.pkl"), "rb",
    ) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "preprocessing/min-max-scaler2.pkl"), "rb"
) as f:
    scaler = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "preprocessing/one_Hot_encoder.pkl"), "rb"
) as f:
    one_Hot_encoder = joblib.load(f)


@st.cache

def preprocessing_data(data, one_hot_enc, scaler):
    
    # For other variables let's use one-hot-encoder
    multi_categorical_variables = ["13_geology"]

    multi_categorical_data = data[multi_categorical_variables]

    multi_categorical_data = one_hot_enc.transform(multi_categorical_data)

    data = data.drop(multi_categorical_variables, axis=1)

    categorical_data = pd.DataFrame(multi_categorical_data.toarray(),columns=one_hot_enc.get_feature_names_out())

    final_data = pd.concat([data,categorical_data], axis=1)

    final_data2 = scaler.transform(final_data.values.reshape(-1,1))
    final_data = pd.DataFrame(final_data2.reshape(-1,15),columns=final_data.columns)

    return final_data


if submit:

    # collect inputs
    input = {
        "13_elevation": elevation,
        "13_slope": slope,
        "13_aspect": aspect,
        "13_placurv": placurv,
        "13_procurv": procurv,
        "13_lsfactor": lsfactor,
        "13_twi": twi,
        "13_sdoif": sdoif,
        "13_geology":geology
    }

    # create a dataframe
    data = pd.DataFrame(input,index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(data=data,                   one_hot_enc=one_Hot_encoder, scaler=scaler)

    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])
    probas = model.predict_proba(transformed_data)
    probability = "{:.2f}".format(float(probas[:, output]))

    # Display results of the NLP task
    st.header("Results")
    if output == 1:
        st.write(
            "The probability of landslide to occur is {} 😔".format(
                probability
            )
        )
    elif output == 0:
        st.write(
            "The probability of landslide NOT to occur is {} 😊".format(
                probability
            )
        )

st.write("Developed with ❤️ by Group1")