from flask import Flask, render_template, request, redirect, session
from flask_mysqldb import MySQL

app = Flask(__name__)
app.secret_key = "secret123"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "somnath"
app.config["MYSQL_PASSWORD"] = "Code@1"
app.config["MYSQL_DB"] = "flask_db"

mysql = MySQL(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    user = request.form["username"].strip()
    pwd = request.form["password"].strip()

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (user, pwd))
    data = cur.fetchone()

    if data:
        session["user"] = user
        return redirect("/dashboard")
    return "Invalid credentials"

@app.route("/dashboard")
def dashboard():
    if "user" in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT subject, grade FROM marks WHERE username=%s", (session["user"],))
        data = cur.fetchall()

        return render_template("dashboard.html", user=session["user"], data=data)

    return redirect("/")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
