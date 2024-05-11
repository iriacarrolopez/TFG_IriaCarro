import json
import tempfile
import requests
from typing import List, Union, Dict, Any
from fastapi import FastAPI, File, Request, Form, Response, UploadFile
from pydantic import BaseModel, Field
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid, datetime
import pandas as pd
from requests.auth import HTTPBasicAuth

csv_file_path = "housing.csv"

# Función para cargar y limpiar los datos del CSV con Pandas
def limpieza_dataset(file_path):
    # Leer el CSV en un DataFrame de Pandas
    df = pd.read_csv(file_path)
    
    # Aquí puedes realizar cualquier limpieza necesaria en los datos del DataFrame
    df = df.dropna()
    
    return df

# Cargar y limpiar los datos del CSV
cleaned_csv_data = limpieza_dataset(csv_file_path)

app = FastAPI()

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
            "@type": "iids:MessageProcessedNotificationMessage",
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

        return response
    
    elif tipo_mensaje == "ids:ArtifactRequestMessage":

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

# @app.post("/data")
# async def data(request: Request):

#     print(request.headers)
    
#     # Lee el contenido de la solicitud como bytes
#     contenido = await request.body()
    
#     print(contenido)

#     # Decodifica el contenido como JSON
#     datos_json = json.loads(contenido)
    
#     # Valida y convierte los datos JSON en un objeto DataModel
#     datos_modelo = DataModel(**datos_json)


#     if datos_modelo.type == "ids:DescriptionRequestMessage":
        
#         if(datos_modelo.requestedElement is not None):

#             return requests.get("https://ecc-provider:8449/api/offeredResource/" + datos_modelo.requestedElement)

#         else:
#             respuesta = requests.get("https://ecc-provider:8449/")

#             print("Respuesta: " + respuesta)

#             print(respuesta.text)

#             return respuesta.json()

#     elif datos_modelo.type == "ids:ContractRequestMessage":

#         return {"mensaje": "Es una petición de contrato"}

#     elif datos_modelo.type == "ids:ContractAgreementMessage":

#         return {"mensaje": "Acuerdo de contrato"}
    
#     elif datos_modelo.type == "ids:ArtifactRequestMessage":

#         return {"mensaje": "Es un ArtifactRequestMessage"}


#     print("tipo: " + datos_modelo.type)

#     return {"message": "Received the request successfully"}


@app.get("/data")
async def get_data():
    
    return {"message": "¡Hola! Esta es una respuesta para la solicitud GET"}