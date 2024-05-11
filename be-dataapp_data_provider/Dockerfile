FROM python:3

RUN pip install fastapi    uvicorn[standard]    requests    python-multipart    requests_toolbelt   pandas  scikit-learn
RUN mkdir /source
ADD main.py /source
ADD housing.csv /source
ADD cert /source/cert
WORKDIR /source
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083", "--ssl-keyfile", "cert/privatekey.pem", "--ssl-certfile", "cert/execution_core_container.cer"]