from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

#flask-------creat the web application
#render template---display an HTML file from the templte folder
#request------reads information sent by the user
#PIL-----python images libray
#images------open the images
#imageOps--------resize the images
#io-------read uploaded images bytes to saves

#imported libaries(pip install flask tensorflow numpy pillow)


app = Flask(__name__)        #create your Flask application

# Load the trained Deep Learning model
model = tf.keras.models.load_model("digit_model.keras")


@app.route("/", methods=["GET", "POST"])

#app.route-----handle the website homepage(GET-----OPEN the website,POST---the user submits the images form)
def home():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please select an image."
        else:
            try:
                # Read the uploaded image
                image = Image.open(
                    io.BytesIO(file.read())
                ).convert("L")            #covert the images into grayscale

                # Resize to MNIST image dimensions
                image = ImageOps.fit(image, (28, 28))

                # MNIST uses white digits on a black background.
                # Invert typical black-on-white uploaded images.
                if np.asarray(image).mean() > 127:
                    image = ImageOps.invert(image)
                    #original pixel       #after inversion
                        # 0(black)            255(white)
                        #  127(grey)
                        # 255(white)
                # Normalize pixel values(0 to 255 pixel value)
                image_array = (
                    np.asarray(image, dtype=np.float32) / 255.0
                )

                # Add batch dimension: (1, 28, 28)   ----(1,28,28,1)
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                # Predict the digit
                probabilities = model.predict(
                    image_array, verbose=0
                )[0]      #index number

                prediction = int(np.argmax(probabilities))
                confidence = round(
                    float(np.max(probabilities)) * 100, 2
                )

            except Exception:
                error = "Could not process this image. Try another."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)