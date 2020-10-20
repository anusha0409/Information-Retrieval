import os
from flask import Flask, render_template,url_for


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",contacts = ['c1', 'c2', 'c3', 'c4', 'c5'],docs=[["Movie 1","Some year", "something else","another attribute"],["Movie 1","Some year", "something else","another attribute"],["Movie 1","Some year", "something else","another attribute"],["Movie 1","Some year", "something else","another attribute"],["Movie 1","Some year", "something else","another attribute"],["Movie 1","Some year", "something else","another attribute"]]);   


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)

#  ,docs=[["Movie 1","Some year", "something else","another attribute"]]