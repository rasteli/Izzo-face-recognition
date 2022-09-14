import flask
import numpy
from flask import Flask, request
import face_recognition as fr

def create_app():
    app = Flask(__name__)
    return app

app = create_app()

def rec_face(url_foto):
    foto = fr.load_image_file(url_foto)
    locations = fr.face_locations(foto)
    faces = fr.face_encodings(foto)

    if (len(faces) > 0):
        return True, faces, locations

    return False, [], []

def infos(code=200, error_message=""):
    if(code == 200):
        error = False
        error_message = ""
    else:
        error = True

        if(error_message == ""):
            msgs = {
                "e400": "No image has been sent for face detection",
                "e404": "There is no one in the picture",
                "e406": "The detected face is not acceptable. Minimum and Maximum size must be respected"
            }

            msg = "There was an undefined error"

            if (f"e{code}" in msgs.keys()):
                msg = msgs[f"e{code}"]

            error_message = msg
    # error, error_message, faces, locations, person_number, code
    return (error, error_message, [], [], 0, code)

@app.route("/")
def root():
    response = flask.jsonify({"message": "Hello, world!"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Obter rosto

@app.route("/faces", methods=["POST"])
def faces():
    error, error_message, faces, locations, person_number, code = infos()

    try:
        minsize = int(request.headers["minsize"]) # Tamanho mínimo recomendado: 100k px²
    except KeyError:
        minsize = 0
    try:
        maxsize = int(request.headers["maxsize"])  # Tamanho máximo recomendado: 400k px²
    except KeyError:
        maxsize = 1024000

    pictures_number = len(list(request.files.values()))

    if(pictures_number > 0):
        for picture in request.files.values():
            try:
                unknown = rec_face(picture)
                print(f"Unknown: {unknown}")
                person_number = len(unknown[1]) if len(unknown[1]) > person_number else person_number

                if (person_number > 0):
                    loc = unknown[2]
                    locations.append(loc)
                    lisDimensions = []

                    for location in loc:
                        width = abs(location[2] - location[0])
                        height = abs(location[1] - location[3])
                        area = width * height
                        lisDimensions.append(area)

                    minimum = 0
                    maximum = 1024000

                    if(len(lisDimensions) != 0):
                        minimum = min(lisDimensions)
                        maximum = max(lisDimensions)

                    if minsize <= minimum and maxsize >= maximum:
                        facesImg = []
                        print(minsize)

                        for f in unknown[1]:
                            face = str(f)
                            face = face.replace("\n", "")
                            face = face.replace("\\", "")

                            while ("  " in face):
                                face = face.replace("  ", " ")

                            face = face.replace(" ", ", ")
                            facesImg.append(face)
                        faces.append(facesImg)
                    else:
                        error, error_message, faces, locations, person_number, code = infos(406)
                        break
                else:
                    error, error_message, faces, locations, person_number, code = infos(404)
                    break
            except Exception as err:
                error, error_message, faces, locations, person_number, code = infos(500, f"ERROR: {err}")
                break
    else:
        error, error_message, faces, locations, person_number, code = infos(400)

    response = flask.jsonify({
        "status": code,
        "error": error,
        "error_message": error_message,
        "faces": str(faces),
        "person_number": person_number,
        "locations": str(locations)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, code

# Comparar rostos

@app.route("/compare", methods=["POST"])
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
            face = numpy.array(eval(face[0]), dtype=float)
            distances = fr.face_distance(familiar_faces, face)
            prob = list(map(lambda x: 1 - x, distances))
            probs.append(prob)

        parcial = []

        for i in range(0, len(probs) - 1):
            for pr in probs:
                print(f"PR: {pr}")
                parcial.append(pr[0])

            media = sum(parcial) / len(parcial)
            res.append(media)
            parcial = []

        index = numpy.argmax(res)
    except Exception as err:
        code = 500
        error = True
        error_message = f"ERROR: {err}"
        print(err)

    response = flask.jsonify({
        "status": code,
        "error": error,
        "error_message": error_message,
        "res": str(res),
        "index": str(index)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, code

if __name__ == "__main__":
    app.run(debug=True, port=8000)
