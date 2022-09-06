import flask
import numpy
from flask import Flask, request
import face_recognition as fr

api = Flask(__name__)

def rec_face(url_foto):
    print(f"Arquivo: {url_foto}")
    
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)

    if (len(rostos) > 0):
        return True, rostos

    return False, []

# Link de como enviar uma imagem diretamente pela API
# LINK: https://cursos.alura.com.br/forum/topico-envio-de-json-e-imagem-em-uma-requisicao-post-com-spring-boot-62481

@api.route("/")
def raiz():
    response = flask.jsonify({"message": "Hello, world!"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Obter rosto

@api.route("/getface", methods=["POST"])
def teste2():
    face = ""
    error = False
    error_message = ""
    code = 400
    faces = []

    for value in request.files.values():
        try:
            desconhecido = rec_face(value)
            
            print(f"Desconhecido: {desconhecido}")
            
            n_pessoas = len(desconhecido[1])

            if (n_pessoas == 1):
                if (desconhecido[0]):
                    rosto = desconhecido[1][0]

                    print(f"Rosto: {rosto}")

                    face = str(rosto)
                    face = face.replace("\n", "")
                    face = face.replace("\\", "")
                    
                    while ("  " in face):
                        face = face.replace("  ", " ")

                    face = face.replace(" ", ", ")
                    faces.append(face)
                    code = 200
            else:
                error = True
                error_message = "There must be one and only one person during face identification"
                code = 400

        except:
            print("There was an error")

    response = flask.jsonify({"status": code, "error": error, "error_message": error_message, 'faces': str(faces)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, code

# Comparar rostos

@api.route("/compare", methods=["POST"])
def compare():
    familiar_faces = numpy.array(eval(request.form["familiar_faces"].replace("array", "")), dtype=float)
    faces = eval(request.form["faces"])

    print(f"Familiar Faces: {familiar_faces}\n")

    probs = []
    code = 200
    error = False
    error_message = ""

    for face in faces:
        face = numpy.array(eval(face), dtype=float)
        print(f"Face in loop: {face}")
        distances = fr.face_distance(familiar_faces, face)
        prob = list(map(lambda x: 1 - x, distances))
        print(f"Distances: {distances}", f"Probs: {probs}")
        probs.append(prob)

    res = []

    media = sum(probs[0])/len(probs[0])
    res.append(media)
    print(f"Media: {media}")
    parcial = []

    index = numpy.argmax(res)
    response = flask.jsonify({
        "status": code, 
        "error": error, 
        "error_message": error_message,
        "probs": str(probs), 
        "res": str(res),
        "index": str(index)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, code


if __name__ == "__main__":
    api.run(debug=True, port=8000)
