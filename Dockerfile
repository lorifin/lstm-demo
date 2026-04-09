FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt streamlit
EXPOSE 8502
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8502", "--server.address=0.0.0.0"]