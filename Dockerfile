# app/Dockerfile Basis

FROM python:3.9-slim

EXPOSE 8501

COPY requirements.txt /app/

WORKDIR /app

RUN pip3 install -r requirements.txt 

COPY streamlit_app.py /app/
COPY example_img/ /app/example_img
COPY serving_model/ /app/serving_model
COPY models/ /app/models
COPY utils /app/utils

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]