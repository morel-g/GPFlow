import dill as pickle


def save_obj(obj, path):
    file = open(path, "wb")
    pickle.dump(obj, file)
    print("- Obj save as " + path)
    file.close()


def load_obj(path):
    file_to_read = open(path, "rb")
    obj = pickle.load(file_to_read)
    print("- Obj load from " + path)
    file_to_read.close()
    return obj
