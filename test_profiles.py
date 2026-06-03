import requests

profiles = [
    # Wet
    {"District":"Cuttack","State":"Odisha","Season":"Kharif","Soil_Type":"Alluvial","Irrigation":"Canal","Rainfall":2200,"Temperature":28,"Humidity":85,"Nitrogen":350,"Phosphorus":150,"Potassium":200,"Soil_pH":6.5,"Area":10000,"Fertilizer":120,"Previous_Yield":1.5,"Year":2026},
    # Dry
    {"District":"Bolangir","State":"Odisha","Season":"Zaid","Soil_Type":"Red","Irrigation":"Rainfed","Rainfall":500,"Temperature":38,"Humidity":40,"Nitrogen":150,"Phosphorus":100,"Potassium":110,"Soil_pH":5.8,"Area":10000,"Fertilizer":50,"Previous_Yield":1.5,"Year":2026},
    # Winter
    {"District":"Angul","State":"Odisha","Season":"Rabi","Soil_Type":"Loamy","Irrigation":"Tube Well","Rainfall":200,"Temperature":18,"Humidity":55,"Nitrogen":280,"Phosphorus":180,"Potassium":210,"Soil_pH":7.0,"Area":10000,"Fertilizer":80,"Previous_Yield":1.5,"Year":2026},
    # Poor soil
    {"District":"Kalahandi","State":"Odisha","Season":"Kharif","Soil_Type":"Laterite","Irrigation":"Rainfed","Rainfall":1100,"Temperature":32,"Humidity":65,"Nitrogen":80,"Phosphorus":50,"Potassium":60,"Soil_pH":4.5,"Area":10000,"Fertilizer":20,"Previous_Yield":1.5,"Year":2026}
]

for p in profiles:
    r = requests.post("http://127.0.0.1:8000/predict", json=p)
    print(r.json()["Best_Crop"], r.json()["Expected_Yield"])
