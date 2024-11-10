# 授权码： qq邮箱-设置
# 接收邮件服务器：imap.qq.com，使用SSL，端口号993
# 发送邮件服务器：smtp.qq.com，使用SSL，端口号465或587
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
 
my_sender=''#你自己的qq邮箱
my_pass = ''#你自己的授权码
my_user=['',]#想发送到的邮箱

def mail(text):
    ret=True
    try:
        msg=MIMEText(text,'plain','utf-8')
        #msg=MIMEText(mail_msg,'html','utf-8')
        msg['From']=formataddr(["期货小助手",my_sender])
        #msg['To']=formataddr(["用户",my_user])
        msg['Subject']="期货牛票推荐" 
        server=smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(my_sender, my_pass)
        server.sendmail(my_sender,my_user,msg.as_string())
        server.quit()
    except Exception:
        ret=False
    return ret
#%%
if __name__ == "__main__":
    ret=mail("NI 方向：1")
    if ret:
        print("邮件发送成功")
    else:
        print("邮件发送失败")
