import smtplib
from email.mime.text import MIMEText
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email import Encoders


#pass_fname = 'pass.txt'
#try:
#    with open(pass_fname, 'r') as f:
#        pas = f.read().strip()
#except:
#    pas = ''

def send_error_email(subject, body):
    to = ['']
    me = ''
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = ", ".join(to)
    pas = ''
    s = smtplib.SMTP('smtp.gmail.com:587')
    s.ehlo()
    s.starttls()
    s.ehlo()
    #s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()

def send_email(subject, body, to):
    to = ['']
    me = ''
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = ", ".join(to)
    pas = ''
    s = smtplib.SMTP('smtp.gmail.com:587')
    s.ehlo()
    s.starttls()
    s.ehlo()
    #s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()

def send_email_wHTML(subject, body, to):
    to = ['']
    me = ''
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = ", ".join(to)
    html_body = MIMEText(body, 'html')
    msg.attach(html_body)
    pas = ''    
    s = smtplib.SMTP('smtp.gmail.com:587')
    s.ehlo()
    s.starttls()
    s.ehlo()
    #s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()


def send_email_attach(subject, body, to, fname):

    me = ''

    msg = MIMEMultipart()

    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = ", ".join(to)

    msg.attach(MIMEText(body))
    pas = ''
    s = smtplib.SMTP('smtp.gmail.com:587')
    s.ehlo()
    s.starttls()
    s.ehlo()
    #s.login(usrnm,pas)
    s.login(me, pas)
    s.sendmail(me, to, msg.as_string())
    s.quit()
