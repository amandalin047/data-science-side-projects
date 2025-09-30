from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
from email.message import EmailMessage
import base64
import os
import mimetypes
import re
from types import SimpleNamespace
from docx import Document

SCOPES = ["https://www.googleapis.com/auth/drive.file",
          "https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/drive",
          "https://www.googleapis.com/auth/script.projects",
          "https://www.googleapis.com/auth/script.external_request",
          "https://www.googleapis.com/auth/script.scriptapp",
          "https://mail.google.com/",
          "https://www.googleapis.com/auth/gmail.readonly",
          "https://www.googleapis.com/auth/gmail.compose",
          "https://www.googleapis.com/auth/gmail.modify",
          "https://www.googleapis.com/auth/gmail.send"]

def authenticate(absolute_directory):
    creds, token_path = None, absolute_directory+"token.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    try:
        return (build("drive", "v3", credentials=creds), build("script", "v1", credentials=creds), build("gmail", "v1", credentials=creds))
    
    except Exception as e:
        os.remove(token_path)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
        return (build("drive", "v3", credentials=creds), build("script", "v1", credentials=creds), build("gmail", "v1", credentials=creds))

def get_draft_body(gmail_service, draft_id, absolute_directory):
    draft = gmail_service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
    message = draft["message"]
    payload = message["payload"]

    texts, images = [], []

    def recursion(part):
        if part.get("mimeType") == "text/plain" and "data" in part.get("body", {}):
            decoded = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
            texts.append(decoded)
        elif part.get("mimeType") == "image/jpeg":
            images.append(part["filename"])
            attachment_id = part["body"]["attachmentId"]
            attachment = gmail_service.users().messages().attachments().get(userId="me",
                                                                            messageId=message["id"],
                                                                            id=attachment_id).execute()
            image_data = base64.urlsafe_b64decode(attachment["data"])
            path_to_save = absolute_directory+part["filename"]
            if not os.path.exists(path_to_save):
                with open(path_to_save, "wb") as f:
                    f.write(image_data)
                
        for subpart in part.get("parts", []):
            recursion(subpart)

    recursion(payload)
    return texts, images


def get_recent_drafts(absolute_directory): 
    results = gmail_service.users().drafts().list(userId="me").execute()
    drafts = results.get("drafts", [])

    now = datetime.now()
    half_hour_ago = now - timedelta(hours=0.5)
    stamp = int(half_hour_ago.timestamp())

    recent_drafts = []
    for draft in drafts:
        draft_id = draft["id"]
        draft_detail = gmail_service.users().drafts().get(userId='me', id=draft_id, format="metadata").execute()
        internal_date_ms = int(draft_detail["message"]["internalDate"]) // 1000

        if internal_date_ms >= stamp:
            texts, images = get_draft_body(gmail_service, draft_detail["id"], absolute_directory)
            text = texts[0].replace("\r", "")
            newspaper = text.split("\n")[0]
            title = text.split("\n")[1]
            content = text.replace(newspaper+"\n"+title+"\n", "")
            obj = SimpleNamespace(title=title,
                                  content=content,
                                  newspaper=newspaper,
                                  images=images,
                                  draft_id=draft_id)      
            recent_drafts.append(obj)  
    return recent_drafts


def get_word_count(text):
    punc = " \n！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    pattern = f"[{re.escape(punc)}]" 
    cleaned_with_punc = re.sub(r"[\n  ]", "", text)
    cleaned_without_punc = re.sub(pattern, "", text)
    count_with_punc = len(re.sub(r"[\n  ]", "", text))
    count_without_punc = len(re.sub(pattern, "", text))   
    return count_with_punc, count_without_punc


def send_email(gmail_service, to, newspaper, plain_text, file_path, html_text=None, image_paths=[]):
    message = EmailMessage()
    message.set_content(plain_text)

    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
    
    subject = f"投稿{newspaper}：{file_name.split('.')[0]}"
    if html_text:
        message.add_alternative(html_text, subtype='html')
    message["To"] = to
    message["From"] = "my.mothers.email@gmail.com"
    message["Subject"] = subject  

    message.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

    if image_paths:
        for image_path in image_paths:
            with open(absolute_directory+image_path, "rb") as img:
                mime_type, _ = mimetypes.guess_type(image_path)
                main_type, sub_type = mime_type.split("/", 1)
                message.add_attachment(img.read(), maintype=main_type, subtype=sub_type, filename=absolute_directory+image_path)
        
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    create_message = {
        "raw": encoded_message
    }

    send_message = gmail_service.users().messages().send(userId="me", body=create_message).execute()
    print(f"已將'{file_name.split('.')[0]}'寄至{newspaper}")
    return send_message


def create_html_text(text, title):
    count_with_punc, count_without_punc = get_word_count(text)
    text = text.replace("\n", "<br>")
    html_text = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title></title>
  </head>
  <body>
    <p><b>標題：</b>{title}</p>
    <p><b>內文全文：</b><br>{text}</p><br>
    <p><b>字數：</b>{count_with_punc}（含標點）/{count_without_punc}（不含標點）</p>
    <p><b>聯絡電話：</b>0900000000</p>
    <p><b>真實姓名：</b>NAME（欲用筆名 name 發表）</p>
  </body>
</html>
""" 
    return html_text

def create_doc(obj, directory):
    file_path_doc = directory + obj.title + ".doc"
    file_path_docx = directory + obj.title + ".docx"
    if os.path.exists(file_path_doc):
        os.remove(file_path_doc)
    elif os.path.exists(file_path_docx):
        os.remove(file_path_docx)

    doc = Document()
    doc.add_paragraph(obj.content)
    file_path = directory + f"{obj.title}.doc"
    try:
        doc.save(file_path)
    except Exception as e:
        print(e)
    #print(f"Created file: {obj.title}")
    return file_path


absolute_directory = "absolute_directory_here/"
newspaper_email_dict = {"TEST": "my.email@gmail.com",
                        "聯合報D家庭副刊": "family@udngroup.com", # "family@udngroup.com"
                        "聯合報D健康": "health@udngroup.com", # "health@udngroup.com"
                        "聯合報繽紛版": "benfen@udngroup.com", #"benfen@udngroup.com",
                        "人間福報副刊": "mtnart7@merit-times.com.tw", # "mtnart7@merit-times.com.tw"
                        "人間福報家庭": "mtnart12@merit-times.com.tw", # "mtnart12@merit-times.com.tw"
                        "自由時報家庭plus": "family@libertytimes.com.tw", # "family@libertytimes.com.tw"
                        "國語日報家庭版": "edit12@mdnkids.com" # "edit12#mdnkids.com"
                        }

if __name__ == "__main__":
    drive_service, script_service, gmail_service = authenticate(absolute_directory)
    recent_drafts = get_recent_drafts(absolute_directory)
    for obj in recent_drafts:
        file_path = create_doc(obj, absolute_directory)
        html_text = create_html_text(obj.content, obj.title)
        try:
            send_email(
            gmail_service,
            newspaper_email_dict[obj.newspaper],
            obj.newspaper,
            obj.content,
            file_path,
            html_text=html_text,
            image_paths=obj.images
            )
        except KeyError:
            print(f"'{obj.title}'寄件草稿格式錯誤，請修改後再試一次。")
        else: 
            gmail_service.users().drafts().delete(userId="me", id=obj.draft_id).execute()
