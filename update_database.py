from ocr_analysis import model_run
import os
import sys


DATABASE_PATH=f"{os.path.abspath(os.path.dirname(__file__))}/database.csv"

#############
def dict_to_string(data, indent=0):
    """Convert a dictionary to a structured string."""
    result = ""
    spaces = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if key=="person_name":
                continue
            result += f"{spaces}{key}: "
            if isinstance(value, (dict, list)):
                result += ", " + dict_to_string(value, indent + 4)
            else:
                result += f"{value}, "
    elif isinstance(data, list):
        for item in data:
            result += f"{spaces}- "
            if isinstance(item, (dict, list)):
                result += ", " + dict_to_string(item, indent + 4)
            else:
                result += f"{item}, "
    else:
        result += f"{spaces}{data}, "
    return result.replace("|",";")
############


def database_insert(person_name, file_type, string_output, key):
    
    if not os.path.isfile(DATABASE_PATH):
        with open(DATABASE_PATH, "a") as f:
            f.write("Person|Document|Details|Key\n")

    with open(DATABASE_PATH, "a") as f:
        f.write(f"{person_name}|{file_type}|{string_output}|{key}\n")

def update(file_type, llm_data, key=0):

    
    person_name=llm_data["person_name"]
    string_output=dict_to_string(llm_data)

    database_insert(person_name, file_type, string_output, key)


def main():
    try:
        file_type=sys.argv[1]
        parse_data=sys.argv[2]
    except:
        file_type=model_run.FILE_TYPE
        parse_data=model_run.PARSE_DATA    
    
    output=model_run.pull_data(file_type, parse_data)
    update(file_type, output)

if __name__ == "__main__":
    main()