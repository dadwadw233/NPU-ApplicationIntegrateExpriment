import pymysql
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import urllib
import urllib.request
import hashlib

def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}

smsapi = "******"

user = 'd******'

password = md5('*****')

def send_message(phone, content):
    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print (statusStr[the_page])
def create_database(name, cur):
    try:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {name}")
        print(f"create database: {name} successfully")
    except Exception as e:
        print(f"error: {e}")


def create_table(name, cur):
    try:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                time DATETIME,
                content TEXT
            )
        """)
        print(f"create table: {name} successfully")
    except Exception as e:
        print(f"error: {e}")

def add_event(time, event, cur, table):
    try:
        cur.execute(f"INSERT INTO `{table}` (`time`, `content`) VALUES (%s, %s)", (time, event))
        print(f"create event successfully")
    except Exception as e:
        print(f"error: {e}")
pymysql_config = {
    'host': '******',
    'port': 63937,
    'user': 'root',
    'password': '*****',
    'charset': 'utf8',
}

conn = pymysql.connect(**pymysql_config)
cur = conn.cursor()

print("是否创建一个新的数据库（建议不要qaq）？（Y/N）")
choice = input()
if choice == "Y" or choice == 'y':
    name = input("输入数据库名称： ")
    create_database(name, cur)

database = input("输入你要使用的数据库： ")
pymysql_config = {
    'host': '******',
    'port': 63937,
    'user': 'root',
    'password': '******',
    'database': database,
    'charset': 'utf8',
}
conn = pymysql.connect(**pymysql_config)
cur = conn.cursor()
print("是否创建一个新表（建议不要qaq）？（Y/N）")
choice = input()

if choice == "Y" or choice == 'y':
    name = input("输入表名： ")
    create_table(name, cur)

table = input("输入你要使用的表： ")
# INSERT INTO `test1` (`id`, `time`, `content`) VALUES (1, '2023-10-05 21:02:43', '11111');
running = True
while running:
    choice = input("选择操作（1：新建待办实事项; 2: 发送所有待办事项到手机; 3:发送所有信息到邮箱; 4：退出（其他的还没写qaq））： ")
    if choice == '1':
        time = input("输入待办事项日期（xxxx-xx-xx xx:xx:xx）: ")
        event = input("输入待办内容： ")
        add_event(time, event, cur, table)
        conn.commit()
    elif choice == '2':
        phone = input("输入手机号（中国大陆）： ")
        cur.execute(f"SELECT * FROM `{table}`")
        records = cur.fetchall()

        formatted_records = ""
        for record in records:
            datetime_str = record[1].strftime("%Y-%m-%d %H:%M:%S")
            content = record[2]
            formatted_records += f"{datetime_str}: {content}\n"
        send_message(phone, formatted_records)

    elif choice == '3':
        email = input("输入邮箱： ")
        cur.execute(f"SELECT * FROM `{table}`")
        records = cur.fetchall()
        sender_email = "******"
        password = "*******"
        formatted_records = ""
        for record in records:
            datetime_str = record[1].strftime("%Y-%m-%d %H:%M:%S")
            content = record[2]
            formatted_records += f"{datetime_str}: {content}\n"
        subject = '待办提醒'
        message = formatted_records
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            # 发送邮件
            server.sendmail(sender_email, email, msg.as_string())
            print('邮件发送成功！')

        except Exception as e:
            print('邮件发送失败:', str(e))

        finally:
            server.quit()
    else:
        break