import requests

resp = requests.post("http://127.0.0.1:5000/", files={'file': open('../testImg/3+/IHC.png', 'rb')})

print(resp.text)
