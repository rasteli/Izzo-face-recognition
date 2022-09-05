import flask
import numpy
from flask import Flask, request
import face_recognition as fr

api = Flask(__name__)

def rec_face(url_foto):
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
    code = 404
    faces = []

    for x in range(1, len(request.files)+1):
        arquivo = request.files.get(f"img{x}")
        
        try:
            desconhecido = rec_face(arquivo)
        except:
            print("There was an error.")
        
        n_pessoas = len(desconhecido[1])

        if (n_pessoas == 1):
            if (desconhecido[0]):
                rosto = desconhecido[1][0]
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

    response = flask.jsonify({"status": code, "error": error, "error_message": error_message, 'faces': str(faces)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, code

# Comparar rostos

@api.route("/compare", methods=["POST"])
def compare():
    familiar_faces = eval(request.form["familiar_faces"])
    faces = eval(request.form["faces"])

    print(familiar_faces)

    probs = []
    code = 200
    error = False
    error_message = ""

    for face in faces:
        face = numpy.array(eval(face), dtype=float)
        distances = fr.face_distance(familiar_faces, face)
        prob = list(map(lambda x: 1 - x, distances))
        probs.append(prob)

    res = []
    parcial = []

    for i in range(0, len(probs[0])-1):
        for pr in probs:
            parcial.append(pr[i])

        media = sum(parcial)/len(parcial)
        res.append(media)
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
