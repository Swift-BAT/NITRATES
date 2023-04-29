import smtplib
import sys
from ..config import PASSWD_FILE
from email.mime.text import MIMEText

if sys.version_info < (3, 0):
    from email.MIMEMultipart import MIMEMultipart
else:
    from email.mime.multipart import MIMEMultipart

if sys.version_info < (3, 0):
    from email.MIMEBase import MIMEBase
else:
    from email.mime.base import MIMEBase

if sys.version_info < (3, 0):
    from email import Encoders
else:
    from email import encoders


pass_fname = PASSWD_FILE  #'pass.txt'
try:
    with open(pass_fname, "r") as f:
        pas = f.read().strip()
except:
    pas = ""


def send_error_email(subject, body):
    to = [
        "gzr5209@psu.edu",
        "sjs8171@psu.edu",
        "jak51@psu.edu",
        "aaron.tohu@gmail.com",
        "delauj2@gmail.com",
    ]
    # to = ['gzr5209@psu.edu', 'sjs8171@psu.edu','delauj2@gmail.com']
    me = "amon.psu@gmail.com"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = me
    msg["To"] = ", ".join(to)
    s = smtplib.SMTP("smtp.gmail.com:587")
    s.ehlo()
    s.starttls()
    s.ehlo()
    # s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()


def send_email(subject, body, to):
    to = [
        "gzr5209@psu.edu",
        "sjs8171@psu.edu",
        "jak51@psu.edu",
        "aaron.tohu@gmail.com",
        "delauj2@gmail.com",
    ]
    # to = ['gzr5209@psu.edu', 'sjs8171@psu.edu','delauj2@gmail.com']
    me = "amon.psu@gmail.com"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = me
    msg["To"] = ", ".join(to)
    s = smtplib.SMTP("smtp.gmail.com:587")
    s.ehlo()
    s.starttls()
    s.ehlo()
    # s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()


def send_email_wHTML(subject, body, to):
    to = [
        "gzr5209@psu.edu",
        "sjs8171@psu.edu",
        "jak51@psu.edu",
        "aaron.tohu@gmail.com",
        "delauj2@gmail.com",
    ]
    # to = ['gzr5209@psu.edu', 'sjs8171@psu.edu','delauj2@gmail.com']
    me = "amon.psu@gmail.com"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = me
    msg["To"] = ", ".join(to)
    html_body = MIMEText(body, "html")
    msg.attach(html_body)
    s = smtplib.SMTP("smtp.gmail.com:587")
    s.ehlo()
    s.starttls()
    s.ehlo()
    # s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()


def send_email_attach(subject, body, to, fname):
    me = "amon.psu@gmail.com"

    msg = MIMEMultipart()

    msg["Subject"] = subject
    msg["From"] = me
    msg["To"] = ", ".join(to)
    msg.attach(MIMEText(body))
    s = smtplib.SMTP("smtp.gmail.com:587")
    s.ehlo()
    s.starttls()
    s.ehlo()
    # s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()
