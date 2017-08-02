from flask import Flask, render_template, json, request
import re, math
import logging

from m3app.m3crnn import m3crnn

# @TODO divide this file into separate applications using FLASK blueprints

app = Flask(__name__)
app.register_blueprint(m3crnn)


@app.route("/")
def main():
	return render_template('about.html')

@app.route("/about")
def about():
	return render_template('about.html')

#--------------------------------
@app.route("/header")
def header():
	return render_template('header.html')

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8080)



