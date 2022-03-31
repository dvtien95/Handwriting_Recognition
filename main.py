from flask import Flask, request, render_template, jsonify
import argparse
import mynet
import numpy as np
import re
import base64

app = Flask(__name__)

def load_model(bin_dir):
    """
    Load model
        Arguments:
            bin_dir: The directory of the bin (normally bin/)
        Returns:
            Loaded model from file
    """
    model = mynet.get_model(bin_dir)
    return model


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    """
    Called when user presses the predict button.
    Processes the canvas and handles the image.
    Passes the loaded image into the neural network, and it makes
    class prediction.
    """

    # Local functions
    def crop(x):
        _len = len(x) - 1
        for index, row in enumerate(x[::-1]):
            z_flag = False
            for item in row:
                if item != 0:
                    z_flag = True
                    break
            if z_flag == False:
                x = np.delete(x, _len - index, 0)
        return x

    def parseImage(imgData):
        # parse canvas bytes and save as output.png
        imgstr = re.search(b'base64,(.*)', imgData).group(1)
        with open('output.png', 'wb') as output:
            output.write(base64.decodebytes(imgstr))

    # get data from drawing canvas and save as image
    parseImage(request.get_data())
    # read parsed image back in 8-bit, black and white mode (L)
    x = mynet.transform_image()

    # Predict from model
    out = mynet.get_prediction(model, x)

    # Generate response
    response = {'prediction1': out[0][0],
                'confidence1': out[1][0],
                'prediction2': out[0][1],
                'confidence2': out[1][1],
                'prediction3': out[0][2],
                'confidence3': out[1][2]
                }

    return jsonify(response)


@app.route("/")
def index():
    """
    Render index for user connecting to /
    """
    return render_template('index.html')


if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(description='A webapp for testing models generated from mynet.py'
                                                 ' on the EMNIST dataset')
    parser.add_argument('--bin', type=str, default='bin',
                        help='Directory to the bin containing the model')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host to run the flask server on')
    parser.add_argument('--port', type=int, default=5000, help='The port to run the flask server on')
    args = parser.parse_args()
    model = load_model(args.bin)
    app.run(host=args.host, port=args.port, debug=True)
