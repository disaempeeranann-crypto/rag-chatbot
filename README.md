# RAG Chatbot (Streamlit)

แชตบอตที่ให้ผู้ใช้อัปโหลดเอกสารเพื่อสร้าง RAG เอง
ถ้าหาคำตอบในเอกสารไม่เจอ จะเรียก LLM ฟรี (Pollinations) ที่ไม่ต้องใช้ API key

## รันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ผ่าน GitHub + Streamlit Community Cloud
1. สร้าง repo ใหม่บน GitHub แล้วอัปโหลด `app.py`, `requirements.txt`, `README.md`
2. ไปที่ https://share.streamlit.io แล้วล็อกอินด้วย GitHub
3. กด **Create app** → เลือก repo / branch `main` / Main file path = `app.py`
4. กด **Deploy**
