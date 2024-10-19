from flask import Flask, send_file

app = Flask(__name__)

@app.route('/plot_image')
def plot_image():
    # Replace 'path/to/your/plot.png' with the actual path to your Matplotlib image
    image_path = r'C:\Users\Shivam Dutt Sharma\Desktop\Prajna\ConversationalAIPrajna\plot.png'
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)