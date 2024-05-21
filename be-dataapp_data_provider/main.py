import json
import tempfile
import requests
from typing import List, Union, Dict, Any
from fastapi import FastAPI, File, Request, Form, Response, UploadFile
from pydantic import BaseModel, Field
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid, datetime
from requests.auth import HTTPBasicAuth
from analitica import *

app = FastAPI()

csv_file_path = "C:/Users/IRIA/Desktop/true-connector/be-dataapp_data_provider/housing.csv"

@app.get("/logisticRegression")
async def logistic_regression():

    X_train, y_train, test = preprocess_data(csv_file_path)
    
    log_reg = logistic_regression_california_housing(X_train, y_train, max_iter=100)

    return {"Logistic regression: ": log_reg}

@app.get("/linearRegression")
async def linear_regression():
    X_train, y_train, test = preprocess_data(csv_file_path)

    reg, error = train_linear_regression_and_evaluate(X_train, y_train)

    return {"reg": reg, "error": error}

@app.get("/decisionTreeRegressor")
async def decision_tree_regressor():
    X_train, y_train, test = preprocess_data(csv_file_path)

    return train_and_evaluate_models(X_train, y_train, cv=10)

@app.get("/randomForestRegressor")
async def random_forest_regressor():
    X_train, y_train, test = preprocess_data(csv_file_path)

    forest_reg, forest_rmse = train_random_forest_and_evaluate(X_train, y_train, cv=10)

    return {"Forest reg": forest_reg, "Forest_rmse": forest_rmse}

@app.get("/gridSearch")
async def grid_search():
    # Llamar a la función para preparar los datos
    X_train, y_train, test = preprocess_data(csv_file_path)
    
    # Llamar a la función para entrenar el modelo
    best_params, best_model = preprocess_pipeline(X_train, y_train)

    return {"best_params": best_params, "best_model": str(best_model)}

@app.get("/proxy")
def proxy(request: Request):
    print(request)
    return {"Hello": "World"}


@app.post("/data")
async def upload_text(header: str = Form(...), payload: str = Form(None)):

    mensaje = json.loads(header)

    print(mensaje)

    print("ID")

    #print(mensaje["ids:requestedElement"]["@id"])

    print(mensaje.keys())

    print("PAYLOAD")

    tipo_mensaje = mensaje["@type"]

    fecha_hora_actual = datetime.datetime.now()

    fecha = fecha_hora_actual.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    basic = HTTPBasicAuth('apiUser', 'passwordApiUser')

    if tipo_mensaje == "ids:DescriptionRequestMessage":


        if "ids:requestedElement" not in mensaje.keys():

            respuesta = requests.get("https://ecc-provider:8449/", verify=False)

        else:

            url = "https://ecc-provider:8449/api/offeredResource/"
            headers = {'resource': mensaje["ids:requestedElement"]["@id"]}

            respuesta = requests.get(url, headers=headers, verify=False, auth=basic)
        
        
        respuesta_ecc = respuesta.json()

        print("RESPUESTA:")

        print(respuesta_ecc)

        # print("ID MENSAJE:")

        # print(mensaje["@id"])

        # print("ID RESPUESTA ECC:")

        # print(respuesta_ecc["@id"])

        text_bytes = json.dumps(respuesta_ecc)

        diccionario = {
            "@context":{
                "ids": "https://w3id.org/idsa/core/",
                "idsc": "https://w3id.org/idsa/code/"
            },
            "@type": "ids:DescriptionResponseMessage",
            "@id": "https://w3id.org/idsa/autogen/descriptionResponseMessage/" + str(uuid.uuid4()),
            "ids:securityToken":{
                "@type": "ids:DynamicAttributeToken",
                "@id": "https://w3id.org/idsa/autogen/dynamicAttributeToken/d599a43f-6538-483e-b069-9d328df38527",
                "ids:tokenValue": "DummyTokenValue",
                "ids:tokenFormat": {
                    "@id": "https://w3id.org/idsa/code/JWT"
                }
            },
            "ids:modelVersion": mensaje["ids:modelVersion"],
            "ids:issued":{
                "@value": fecha,
                "@type": "http://www.w3.org/2001/XMLSchema#dateTimeStamp"
            },
            "ids:issuerConnector":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:senderAgent":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:correlationMessage":{
                "@id": mensaje["@id"]
            },
            "ids:recipientConnector":[
                mensaje["ids:issuerConnector"]
            ]
        }

        print("DICCIONARIO")

        print(diccionario)
        
        str_header = json.dumps(diccionario)

        multipart_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(str_header.encode('utf-8'))) + "\n\n"

        payload_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(text_bytes.encode('utf-8'))) + "\n\n" 

        # Create a multipart encoder
        encoder = MultipartEncoder(
            fields={
                'header': multipart_header + str_header,
                'payload':  payload_header + text_bytes
            }
        )

        #print(encoder.to_string()) 

        # Create the response with the multipart data
        response = Response(content=encoder.to_string(), media_type=encoder.content_type)

        # print(response)
        
        return response
    
    elif tipo_mensaje == "ids:ContractRequestMessage":

        print("PAYLOAD")

        print(json.loads(payload))

        str_payload = json.loads(payload)

        print("ID PAYLOAD")

        print(str_payload["@id"])

        url = "https://ecc-provider:8449/api/contractOffer/"
        headers = {'contractOffer': str_payload["@id"]}

        respuesta = requests.get(url, headers=headers, verify=False, auth=basic)
        
        #respuesta = requests.get("https://ecc-provider:8449/api/contractOffer/", verify=False, auth=basic)

        respuesta_ecc = respuesta.json()

        print("RESPUESTA ECC:")
        
        print(respuesta_ecc)

        text_bytes = json.dumps(respuesta_ecc)

        id_aleatorio = str(uuid.uuid4())

        diccionario = {
            "@context":{
                "ids": "https://w3id.org/idsa/core/",
                "idsc": "https://w3id.org/idsa/code/"
            },
            "@type": "ids:ContractAgreementMessage",
            "@id": "https://w3id.org/idsa/autogen/contractAgreementMessage/" + id_aleatorio,
            "ids:issued":{
                "@value": fecha,
                "@type": "http://www.w3.org/2001/XMLSchema#dateTimeStamp"
            },
            "ids:issuerConnector":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:senderAgent":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:securityToken":{
                "@type": "ids:DynamicAttributeToken",
                "@id": "https://w3id.org/idsa/autogen/dynamicAttributeToken/d599a43f-6538-483e-b069-9d328df38527",
                "ids:tokenValue": "DummyTokenValue",
                "ids:tokenFormat": {
                    "@id": "https://w3id.org/idsa/code/JWT"
                }
            },
            "ids:modelVersion": mensaje["ids:modelVersion"],
            "ids:correlationMessage":{
                "@id": mensaje["@id"]
            },
            "ids:recipientConnector":[ 
                mensaje["ids:issuerConnector"]
            ]
        }
        
        str_header = json.dumps(diccionario)

        multipart_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(str_header.encode('utf-8'))) + "\n\n"

        payload_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(text_bytes.encode('utf-8'))) + "\n\n" 

        # Create a multipart encoder
        encoder = MultipartEncoder(
            fields={
                'header': multipart_header + str_header,
                'payload':  payload_header + text_bytes
            }
        )

        response = Response(content=encoder.to_string(), media_type=encoder.content_type)

        return response
    
    elif tipo_mensaje == "ids:ContractAgreementMessage":

        print("PAYLOAD")

        print(json.loads(payload))

        str_payload = json.loads(payload)

        print("ID PAYLOAD")

        print(str_payload["@id"])

        # url = "https://ecc-provider:8449/api/contractOffer/"
        # headers = {'contractOffer': str_payload["@id"]}

        # respuesta = requests.get(url, headers=headers, verify=False, auth=basic)
        
        respuesta = requests.get("https://ecc-provider:8449/", verify=False)

        respuesta_ecc = respuesta.json()

        print("RESPUESTA ECC:")
        
        print(respuesta_ecc)

        text_bytes = json.dumps(respuesta_ecc)

        id_aleatorio = str(uuid.uuid4())

        diccionario = {
            "@context":{
                "ids": "https://w3id.org/idsa/core/",
                "idsc": "https://w3id.org/idsa/code/"
            },
            "@type": "ids:MessageProcessedNotificationMessage",
            "@id": "https://w3id.org/idsa/autogen/messageProcessedNotificationMessage/" + id_aleatorio,
            "ids:modelVersion": mensaje["ids:modelVersion"],
            "ids:correlationMessage":{
                "@id": mensaje["@id"]
            },
            "ids:recipientConnector":[ 
                mensaje["ids:issuerConnector"]
            ],
            "ids:issued":{
                "@value": fecha,
                "@type": "http://www.w3.org/2001/XMLSchema#dateTimeStamp"
            },
            "ids:issuerConnector":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:senderAgent":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:securityToken":{
                "@type": "ids:DynamicAttributeToken",
                "@id": "https://w3id.org/idsa/autogen/dynamicAttributeToken/d599a43f-6538-483e-b069-9d328df38527",
                "ids:tokenValue": "DummyTokenValue",
                "ids:tokenFormat": {
                    "@id": "https://w3id.org/idsa/code/JWT"
                }
            }
        }

        str_header = json.dumps(diccionario)

        multipart_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(str_header.encode('utf-8'))) + "\n\n"

#         payload_header = '''Content-Type: application/ld+json
# Content-Transfer-Encoding: 8bit
# Content-Length: ''' + str(len(text_bytes.encode('utf-8'))) + "\n\n" 

        # Create a multipart encoder
        encoder = MultipartEncoder(
            fields={
                'header': multipart_header + str_header
            }
        )

        response = Response(content=encoder.to_string(), media_type=encoder.content_type)

        print(response.body)

        return response
    
    elif tipo_mensaje == "ids:ArtifactRequestMessage":

        respuesta = requests.get("https://ecc-provider:8449/", verify=False)

        respuesta_ecc = respuesta.json()

        print("RESPUESTA ECC:")
        
        print(respuesta_ecc)

        text_bytes = json.dumps(respuesta_ecc)

        id_aleatorio = str(uuid.uuid4())

        # if metodo == "decisionTree":
        #     decisionTree(parametros)

        diccionario = {
            "@context":{
                "ids": "https://w3id.org/idsa/core/",
                "idsc": "https://w3id.org/idsa/code/"
            },
            "@type": "ids:ArtifactResponseMessage",
            "@id": "https://w3id.org/idsa/autogen/artifactResponseMessage/" + id_aleatorio,
            "ids:modelVersion": mensaje["ids:modelVersion"],
            "ids:transferContract":{
                "@id": mensaje["ids:transferContract"]["@id"]
            },
            "ids:correlationMessage":{
                "@id": mensaje["@id"]
            },
            "ids:recipientConnector":[ 
                mensaje["ids:issuerConnector"]
            ],
            "ids:issued":{
                "@value": fecha,
                "@type": "http://www.w3.org/2001/XMLSchema#dateTimeStamp"
            },
            "ids:issuerConnector":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:senderAgent":{
                "@id": respuesta_ecc["@id"]
            },
            "ids:securityToken":{
                "@type": "ids:DynamicAttributeToken",
                "@id": "https://w3id.org/idsa/autogen/dynamicAttributeToken/d599a43f-6538-483e-b069-9d328df38527",
                "ids:tokenValue": "DummyTokenValue",
                "ids:tokenFormat": {
                    "@id": "https://w3id.org/idsa/code/JWT"
                }
            }
        }

        str_header = json.dumps(diccionario)

        multipart_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(str_header.encode('utf-8'))) + "\n\n"

        payload_header = '''Content-Type: application/ld+json
Content-Transfer-Encoding: 8bit
Content-Length: ''' + str(len(text_bytes.encode('utf-8'))) + "\n\n" 

        # Create a multipart encoder
        encoder = MultipartEncoder(
            fields={
                'header': multipart_header + str_header,
                'payload':  payload_header + text_bytes
            }
        )

        response = Response(content=encoder.to_string(), media_type=encoder.content_type)

        return response
    
    

    return {"processed_data": header}


@app.get("/data")
async def get_data():
    
    return {"message": "¡Hola! Esta es una respuesta para la solicitud GET"}