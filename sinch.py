import requests

servicePlanId = "c3262b4a85ee40e298b3b8500db80cae"
apiToken = "057bf0ae37c441379f7428b52b76b4fd"
sinchNumber = "447520652769"
toNumber = "8613809008267"
url = "https://us.sms.api.sinch.com/xms/v1/" + servicePlanId + "/batches"

payload = {
  "from": sinchNumber,
  "to": [
    toNumber
  ],
  "body": "Verification Code: 9537. Training Complete!"
}

headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer " + apiToken
}

response = requests.post(url, json=payload, headers=headers)

data = response.json()
print(data)

