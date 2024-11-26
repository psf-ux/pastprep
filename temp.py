import requests

# Define the URL of your Flask API
url = 'http://127.0.0.1:5000/query'  # Replace with your actual API endpoint

# Define the payload (data to be sent in the POST request)
payload = {
    "question": "give me all questions related to migration in islamiyat",
    "category": "olevels",
    "course": "Islamiyat"
}

# Send a POST request to the /query endpoint
response = requests.post(url, json=payload)

# Check the response status
if response.status_code == 200:
    # If successful, print the response JSON
    print("Response JSON:", response.json())
else:
    print("Failed to get a response. Status Code:", response.status_code)
    print("Error Message:", response.text)
