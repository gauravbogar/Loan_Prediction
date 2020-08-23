import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request, jsonify

app = Flask(__name__)
current_path = os.getcwd()
pickle_path = os.path.join(current_path, "assets", "loan.pkl")
classifier = pickle.load(open(pickle_path, "rb"))
data_path = os.path.join(current_path, "assets", "loan.csv")
df = pd.read_csv(data_path)
address_state = df["addr_state"].value_counts().index
house_owner = df["home_ownership"].value_counts().index
purpose_loan = df["purpose"].value_counts().index


@app.route("/")
@app.route("/home")
def customer_details():
    return render_template(
        "index.html",
        address_state=address_state,
        house_owner=house_owner,
        purpose_loan=purpose_loan,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.form
    loan_amnt = data["loan_amnt"]
    term = data["term"]
    int_rate = data["int_rate"]
    emp_length = data["emp_length"]
    home_ownership = data["home_ownership"]
    annual_inc = data["annual_inc"]
    purpose = data["purpose"]
    addr_state = data["addr_state"]
    dti = data["dti"]
    delinq_2yrs = data["delinq_2yrs"]
    revol_util = data["revol_util"]
    total_acc = data["total_acc"]
    longest_credit_length = data["longest_credit_length"]
    verification_status = data["verification_status"]
    cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "purpose",
        "addr_state",
        "dti",
        "delinq_2yrs",
        "revol_util",
        "total_acc",
        "longest_credit_length",
        "verification_status",
    ]
    test_data = pd.DataFrame(
        [
            [
                loan_amnt,
                term,
                int_rate,
                emp_length,
                home_ownership,
                annual_inc,
                purpose,
                addr_state,
                dti,
                delinq_2yrs,
                revol_util,
                total_acc,
                longest_credit_length,
                verification_status,
            ]
        ],
        columns=cols,
    )
    pred = classifier.predict(test_data)
    if pred == 0:

        return render_template(
            "index.html", prediction_text="The Loan will not be a DEFAULTER"
        )

    return render_template(
        "index.html", prediction_text="The Loan will be a DEFAULTER", pred=pred
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    loan_amnt = data["loan_amnt"]
    term = data["term"]
    int_rate = data["int_rate"]
    emp_length = data["emp_length"]
    home_ownership = data["home_ownership"]
    annual_inc = data["annual_inc"]
    purpose = data["purpose"]
    addr_state = data["addr_state"]
    dti = data["dti"]
    delinq_2yrs = data["delinq_2yrs"]
    revol_util = data["revol_util"]
    total_acc = data["total_acc"]
    longest_credit_length = data["longest_credit_length"]
    verification_status = data["verification_status"]
    cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "purpose",
        "addr_state",
        "dti",
        "delinq_2yrs",
        "revol_util",
        "total_acc",
        "longest_credit_length",
        "verification_status",
    ]
    test_data = pd.DataFrame(
        [
            [
                loan_amnt,
                term,
                int_rate,
                emp_length,
                home_ownership,
                annual_inc,
                purpose,
                addr_state,
                dti,
                delinq_2yrs,
                revol_util,
                total_acc,
                longest_credit_length,
                verification_status,
            ]
        ],
        columns=cols,
    )
    pred = classifier.predict(test_data)

    if pred == 0:
        return jsonify("Not a Defaulter")
    return jsonify("Defaulter")


if __name__ == "__main__":
    app.run(debug=True)
