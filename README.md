# Password-Analyser
it uses a data of more than 1 million frequently used passwords to analyse your entered password
<<<<<<< HEAD
<<<<<<< HEAD


🔐 Password Commonness & Uniqueness Checker
Overview

This project provides a data-driven password evaluation tool that checks how common or unique a password is based on real-world leaked password datasets. Unlike traditional password strength meters that rely on simple rules (length, symbols, uppercase), this system:

Checks if a password exists in a large dataset of leaked passwords.

Uses a machine learning model to predict commonness for unseen passwords.

Provides a uniqueness score to guide users toward stronger, safer passwords.

It also includes a Tkinter-based GUI for instant feedback on any password.

Features

✅ Exact Match Check: Quickly determines if a password exists in the dataset.

✅ Predicted Commonness: For new passwords, the model estimates the probability of being common.

✅ Uniqueness Score: Gives a numeric score indicating password rarity.

✅ User-Friendly GUI: Interactive interface for easy password testing.

✅ Data-Driven Security: Leverages real leaked datasets rather than just heuristic rules.

How It Works

Data Loading:
Load a large password dataset (e.g., RockYou) into a frequency dictionary.

Model Training:

Extract character n-gram features from the dataset.

Train a logistic regression model to predict if a password is “common” or “uncommon.”

Password Evaluation:

If the password exists in the dataset → return exact frequency and normalized score.

If the password is new → use the trained model to predict commonness.

Compute a uniqueness score combining frequency/prediction for intuitive understanding.

GUI Interaction:
Users type a password in the Tkinter window to instantly see the results.

Installation

Clone the repository:

git clone https://github.com/yourusername/Password-Commonness-Checker.git
cd Password-Commonness-Checker


Create a virtual environment and activate it:

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Make sure the following files are present:

password_model.pkl → Trained ML model

vectorizer.pkl → Feature vectorizer for new passwords

freq.pkl → Frequency dictionary of passwords

Usage

Run the GUI:

python newpasswordUI.py


Enter a password in the input field.

Click Check Password.

View commonness, normalized frequency, and uniqueness score.

The GUI also indicates if a password is very common or fairly unique.

Project Structure
Password-Commonness-Checker/
│
├─ newpasswordUI.py       # Main GUI application
├─ password_model.pkl     # Pre-trained ML model
├─ vectorizer.pkl         # Feature vectorizer
├─ freq.pkl               # Password frequency dictionary
├─ requirements.txt       # Python dependencies
├─ README.md              # Project documentation
└─ dataset/               # Optional folder for raw password dataset

Screenshots

(Add screenshots of your Tkinter GUI here for a more attractive README)

References

Bonneau, J., et al. Password Cracking and Real-World Password Strength. IEEE Security & Privacy, 2012.

RockYou Dataset (2009 Leak) – widely used for password research.

Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR, 2011.

Tkinter Documentation – Python GUI library
=======
>>>>>>> d44fa9305150b9a0e02a78fbb0074c3549c453f2
=======
>>>>>>> 905b6e6c9f4a0caac20cb389db2fc3e785118f38
