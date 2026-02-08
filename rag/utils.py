import tempfile
import os

def save_temp_image(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(data)
    tmp.close()
    return tmp.name
