import time
import telegram
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class SendLog(object):
    def __init__(self, config):
        self.config = config

        self.server_email = smtplib.SMTP_SSL('smtp.googlemail.com', 465)
        self.server_email.ehlo()

        self.server_email.login(self.config['email']['username'], self.config['email']['password'])

        self.sender_email = self.config['email']['username']
        self.chat_id = self.config['telegram']['chat_id']

        self.retries_email = 5
        
        self.subject = "Checkin system | Notice of check-in"

    def send_telegram_message(self, message, telegram_info=None):
        try:
            if telegram_info is None:
                telegram_notify = telegram.Bot(self.config['telegram']['bot'])
                telegram_notify.send_message(chat_id=self.chat_id,
                                                  text=message,
                                                  parse_mode='Markdown')
            else:
                telegram_notify = telegram.Bot(telegram_info['bot'])
                telegram_notify.send_message(chat_id=telegram_info['chat_id'],
                                                  text=message,
                                                  parse_mode='Markdown')
        except Exception as ex:
            print(ex)
        return

    def send_email(self, receiver_email, name, employee_code, checkin_time):
        t1 = time.time()
        msg = MIMEMultipart("alternative")

        msg["Subject"] = self.subject
        msg["From"] = "Checkin System📍"
        msg["To"] = receiver_email

        email_text = "📍 Hello Sir/Madam %s" % name + \
            "\n-------------------" + \
            "\n👤 Employee code: %s" % employee_code + \
            "\n💌 Email: %s" % receiver_email + \
            "\n🏦 Full name: %s" % name + \
            "\n⏰ Attendance time: 📅 %s 🕒 %s" % (checkin_time.split(' ')[0].strip(),
                                                    checkin_time.split(' ')[1].strip()) + \
            "\n Have a nice day!"

        body = MIMEText(email_text, "plain", "utf-8")

        msg.attach(body)

        for i in range(self.retries_email):
            try:
                self.server_email.sendmail(self.sender_email, receiver_email, msg.as_string())

                break
            except:
                self.server_email = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                self.server_email.ehlo()

                self.server_email.login(self.config['email']['username'], self.config['email']['password'])
        print('Sendmail processing time', time.time() - t1)
        
        if receiver_email == self.config['telegram_special']['email_user']:
            self.send_telegram_message(email_text, self.config['telegram_special'])

        return
