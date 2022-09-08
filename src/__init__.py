import flask
import numpy
from flask import Flask, request
import face_recognition as fr

api = Flask(__name__)

def rec_face(url_foto):
    print(f"Arquivo: {url_foto}")

    foto = fr.load_image_file(url_foto)
    locations = fr.face_locations(foto)
    rostos = fr.face_encodings(foto)
    print(locations)

    if (len(rostos) > 0):
        return True, rostos, locations

    return False, [], []

@api.route("/")
def raiz():
    response = flask.jsonify({"message": "Hello, world!"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Obter rosto

@api.route("/faces", methods=["POST"])
def faces():
    error = False
    error_message = ""
    code = 400
    faces = []
    person_number = 0
    locations = []

    for value in request.files.values():
        try:
            unknown = rec_face(value)
            person_number = len(unknown[1])

            if (person_number > 0):
                locations = unknown[2]
                rostos = unknown[1]
                face = str(rostos)
                face = face.replace("\n", "")
                face = face.replace("\\", "")
                while ("  " in face):
                    face = face.replace("  ", " ")

                faces.append(face)
                code = 200
            else:
                error = True
                error_message = "There is no one in the picture"
                code = 400

        except Exception as err:
            error = True
            error_message = f"ERROR: {err}"
            code = 500
            print("There was an internal error")

    response = flask.jsonify({"status": code, "error": error, "error_message": error_message, 'faces': str(faces),
                              "person_number": person_number, "locations": str(locations), "conectors": str(list(zip(locations, faces)))})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, code


# Comparar rostos

@api.route("/compare", methods=["POST"])
def compare():
    index = -1
    res = []
    probs = []
    code = 200
    error = False
    error_message = ""

    try:
        familiar_faces = numpy.array(eval(request.form["familiar_faces"].replace("array", "")), dtype=float)
        faces = eval(request.form["faces"])

        for face in faces:
            face = numpy.array(eval(face), dtype=float)
            distances = fr.face_distance(familiar_faces, face)
            prob = list(map(lambda x: 1 - x, distances))
            probs.append(prob)


        parcial = []
        for i in range(0, len(probs[0])-1):
            for pr in probs:
                parcial.append(pr[i])
            media = sum(parcial)/len(parcial)
            res.append(media)
            parcial = []

        index = numpy.argmax(res)
    except Exception as err:
        code = 500
        error = True
        error_message = f"ERROR: {err}"
    response = flask.jsonify({
        "status": code,
        "error": error,
        "error_message": error_message,
        #"probs": str(probs),
        "res": str(res),
        "index": str(index)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, code


if __name__ == "__main__":
    api.run(debug=True, port=8000)
