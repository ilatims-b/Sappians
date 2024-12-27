def get_format(file_type):
    if file_type in ["aadhar", "aadhar_2"]:
        return aadhaar
    elif file_type=="pan":
        return pan
    else:
        return other


aadhaar = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": { "person_name": {"type": "string"}, "aadhaar_number": {"type": "string"}, "address": {"type": "string"}},
        "required": ["person_name", "aadhaar_number", "address"],
    },
}

pan = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"person_name": {"type": "string"}, "pan_number": {"type": "string"}, "address": {"type": "string"}},
        "required": ["person_name", "pan_number", "address"],
    },
}

other = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"person_name": {"type": "string"}},
        "required": ["person_name"],
    },
}