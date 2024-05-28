import json
import tempfile
import requests
from typing import List, Union, Dict, Any
from fastapi import FastAPI, File, Request, Form, Response, UploadFile
from pydantic import BaseModel, Field
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid, datetime
from urllib.parse import urlparse, parse_qs
from requests.auth import HTTPBasicAuth
from analitica import *

app = FastAPI()

csv_file_path = "C:/Users/IRIA/Desktop/true-connector/be-dataapp_data_provider/housing.csv"

# @app.get("/logisticRegression")
# async def logistic_regression():

#     X_train, y_train, test = preprocess_data(csv_file_path)
    
#     log_reg = logistic_regression_california_housing(X_train, y_train, max_iter=100)

#     return {"Logistic regression: ": log_reg}

# @app.get("/linearRegression")
# async def linear_regression():
#     X_train, y_train, test = preprocess_data(csv_file_path)

#     reg, error = train_linear_regression_and_evaluate(X_train, y_train)

#     return {"reg": reg, "error": error}

# @app.get("/decisionTreeRegressor")
# async def decision_tree_regressor():
#     X_train, y_train, test = preprocess_data(csv_file_path)

#     return train_and_evaluate_models(X_train, y_train, cv=10)

# @app.get("/randomForestRegressor")
# async def random_forest_regressor():
#     X_train, y_train, test = preprocess_data(csv_file_path)

#     forest_reg, forest_rmse = train_random_forest_and_evaluate(X_train, y_train, cv=10)

#     return {"Forest reg": forest_reg, "Forest_rmse": forest_rmse}

# @app.get("/gridSearch")
# async def grid_search():
#     # Llamar a la función para preparar los datos
#     X_train, y_train, test = preprocess_data(csv_file_path)
    
#     # Llamar a la función para entrenar el modelo
#     best_params, best_model = preprocess_pipeline(X_train, y_train)

#     return {"best_params": best_params, "best_model": str(best_model)}

# @app.get("/proxy")
# def proxy(request: Request):
#     print(request)
#     return {"Hello": "World"}


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

        url = mensaje["ids:requestedArtifact"]["@id"]

        # Parsear la URL
        parsed_url = urlparse(url)

        # Obtener la parte del path y extraer el valor en la posición 'gridSearch'
        path_segments = parsed_url.path.split('/')
        metodo = path_segments[3]

        # Parsear los parámetros de la query string en un diccionario
        query_params = parse_qs(parsed_url.query)
        parametros = {key: value[0] for key, value in query_params.items()}

        if metodo == "logisticRegression":

            penalty = parametros.get("penalty", "l2")
            dual = parametros.get("dual", False)
            tol = parametros.get("tol", "1e-4")
            C = parametros.get("c", 1)
            fit_intercept = parametros.get("fit_intercept", True)
            intercept_scaling = parametros.get("intercept_scaling", 1)
            class_weight = parametros.get("class_weight", None)
            random_state = parametros.get("random_state", None)
            solver = parametros.get("solver", "lbfgs")
            max_inter = parametros.get("max_inter", 100)
            multi_class = parametros.get("multi_class", "auto")
            verbose = parametros.get("verbose", 0)
            warm_start = parametros.get("warm_start", False)
            n_jobs = parametros.get("n_jobs", None)
            l1_ratio = parametros.get("l1_ratio", None)

            log_reg = LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_inter, multi_class, verbose, warm_start, l1_ratio)

        elif metodo == "linearRegression":

            fit_intercept = parametros.get("fit_intercept", True)
            copy_X = parametros.get("copy_X", True)
            n_jobs = parametros.get("n_jobs", None)
            positive = parametros.get("positive", False)

            reg = LinearRegression(fit_intercept, copy_X, n_jobs, positive)

        elif metodo == "decisionTreeRegressor":

            criterion = parametros.get("criterion", "squared_error")
            splitter = parametros.get("splitter", "best")
            max_depth = parametros.get("max_depth", None)
            min_samples_split = parametros.get("min_samples_leaf", 2)
            min_samples_leaf = parametros.get("min_samples_leaf", 1)
            min_weight_fraction_leaf = parametros.get("min_weight_fraction_leaf", 1)
            max_features = parametros.get("max_features", None)
            random_state = parametros.get("random_state", None)
            max_leaf_nodes = parametros.get("max_leaf_nodes", None)
            min_impurity_decrease = parametros.get("min_impurity_decrease", 0.0)
            ccp_alpha = parametros.get("ccp_alpha", 0.0)
            monotonic_cst = parametros.get("monotonic_cst", None)

            tree_reg = DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, ccp_alpha, monotonic_cst)

        elif metodo == "randomForestRegressor":
            print("randomForestRegressor")
        elif metodo == "gridSearch":
            print("gridSearch")



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