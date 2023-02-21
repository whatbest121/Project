ไฟล์ Sentiment คือฟังก์ชั้นแรก โดยมี INPUT 
model.predict(vec.transform(['Changes in non-farm payrolls increase.']))

Changes in non-farm payrolls increase. คือ ชื่อรับ INPUT
และเมื่อรันคำสั่งนี้จะแสดง OUTPUT ออกมา 

ไฟล์ Auto-Summarize an article 

scraped_data = urllib.request.urlopen('https://www.clevelandfed.org/en/newsroom-and-events/speeches/sp-20221011-an-update-on-the-economy-and-monetary-policy.aspx')#รับINPUT
บรรทัดที่6 คือจะรับ INPUT เป็น URL 

บรรทัดที่ 51 บรรทัดสุดท้ายจะเป็นการแสดง Output
