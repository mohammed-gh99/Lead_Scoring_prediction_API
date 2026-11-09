import requests

# Local API
# url = 'http://localhost:9696/predict'

# AWS URL used during the successful deployment.
# The Elastic Beanstalk environment was terminated after testing,
# so redeploy the application before using this URL.
url = 'http://lead-scoring-prediction-env.eba-xm3pqq2m.eu-north-1.elasticbeanstalk.com/predict'

customer = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "africa",
    "annual_income": 50000,
    "number_of_courses_viewed": 5,
    "interaction_count": 10,
    "lead_score": 80
}

response = requests.post(url, json=customer)
response.raise_for_status()

result = response.json()

print('Response:', result)

if result['converted']:
    print('Prioritize lead for sales follow-up')
else:
    print('Add lead to nurturing campaign')
