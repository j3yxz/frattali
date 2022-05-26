from subprocess import call
from os import path
from sys import argv
import json

#simple read the json and will return the content of it
def read_json(file_name):

    #tutta sta cosa è da fare mooolto meglio (su linux devo mettere il separatore giusto)
    if file_name.startswith("./"):
        file_name = path.abspath(path.curdir) + file_name.replace("./", "/")
        print(file_name)
        print("-------------------------")

    with open(file_name, 'r') as f:
        print("apre file")
        return json.load(f)

#function that will interpret the json and pass the correct arguments to "call_py" function
def prepare_args(j_arr,fr_type):
    args = []
    for i in j_arr[fr_type]:
        args.append( j_arr[fr_type] [i] )
    generics = args[:4]
    print(generics)
    specs = []
    for i in args[4]:
        specs.append(args[4][i])

    print(specs)
    return [*generics ,*specs ]

#function that will call the correct fractal to be computed
def call_py(argv):
    path, others =argv[0],argv[1:]
    print(path)
    call(["python3", path, *others])







if __name__ == '__main__':
    if len(argv) > 1:
        json_config = argv[1]
        fr_type="mandelbrot"
        if len(argv)>2:
            fr_type = argv[2]
        config = read_json(json_config)
        call_py(prepare_args(config,fr_type))
    else :
        print("No config file provided, insert config file name")
        json_path = input()
        #check if the path is valid and is a file
        if json_path.endswith(".json") and path.isfile(path):
            print("Insert type (available : mandelbrot or julia)")
            json_path_and_type = input()
            if fr_type==None:
                fr_type = "mandelbrot"
            config = read_json(path)
            call_py(prepare_args(config,fr_type))
        else :
            print("File does not exist, using default config")
            config = read_json("./json_example.json")
            call_py(prepare_args(config, "mandelbrot"))
