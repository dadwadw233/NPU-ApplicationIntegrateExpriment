import imaplib
import email
from email.header import decode_header
import time

def get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()
    return body

def check_email(server, username, password):
    try:
        imap_server = imaplib.IMAP4_SSL(server)
        imap_server.login(username, password)
        imap_server.select("inbox")
        result, data = imap_server.search(None, "UNSEEN")
        email_ids = data[0].split()
        unread_count = len(data[0].split())
        print(f"未读邮件数量：{unread_count}")
        if email_ids:
            latest_email_id = email_ids[-1]
            result, msg_data = imap_server.fetch(latest_email_id, "(RFC822)")
            raw_email = msg_data[0][1]

            msg = email.message_from_bytes(raw_email)
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8")
            from_ = msg.get("From")
            email_body = get_email_body(msg)
            print(f"最新邮件主题：{subject}")
            print(f"发件人：{from_}")
            print("邮件正文：")
            #print(email_body)
        else:
            print("没有未读邮件")

        # 关闭连接
        imap_server.logout()

    except Exception as e:
        print(f"检查邮件时出错：{str(e)}")


# 用户输入邮箱账号信息
server = 'imap.gmail.com'
username = 'yuanhongyu.me@gmail.com'
password = '*****'

# 每隔一段时间检查一次
while True:
    check_email(server, username, password)
    #time.sleep(0.1)  # 等待1小时后再次检查
