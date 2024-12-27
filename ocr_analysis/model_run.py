from llama_cpp import Llama
import os
from . import json_formats
import json
import sys



DEBUG=False

FILE_TYPE=""
PARSE_DATA=""

try:
    llm = Llama(
          model_path=f"{os.path.abspath(os.path.dirname(__file__))}/model.gguf",
          # n_gpu_layers=-1, # Uncomment to use GPU acceleration
          # seed=1337, # Uncomment to set a specific seed
          n_ctx=8192, # Uncomment to increase the context window
          verbose=DEBUG,
    )
except:
    print("Model not found")
    raise(FileNotFoundError)
    


def get_response_format(file_type):
    return json_formats.get_format(file_type)
    

def get_output(parse_data, json_format):
    output=llm.create_chat_completion(
        messages=[
    #        {
    #            "role": "system",
    #            "content": "You are a helpful assistant that deciphers OCR data outputs in JSON.",
    #        },
            {"role": "user", "content": parse_data},
        ],
        response_format=json_format,
        temperature=0.7,
        max_tokens=None,
    )
    return output


def pull_data(file_type, parse_data):
    json_format=get_response_format(file_type)
    raw_output=get_output(parse_data, json_format)
    if DEBUG:
        print(raw_output)

    return json.loads(raw_output["choices"][0]["message"]["content"])
        


def main():
    try:
        file_type=sys.argv[1]
        parse_data=sys.argv[2]
    except:
        file_type=FILE_TYPE
        parse_data=PARSE_DATA    
    
    print(pull_data(file_type, parse_data))




if __name__ == "__main__":
    main()
