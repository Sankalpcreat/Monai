import requests

class TextPipeline:
    def __init__(self,model_name='llama3.2:latest',host='localhost',port=11434):
        self.model_name=model_name
        self.app_url=f'http://{host}:{port}/api/generate'

    def process_symptoms(self,symptoms:str):

        prompt=f"Analyze the following symptoms and suggest possible conditions:{symptoms}"
        payload={
            'model':self.model_name,
            'prompt':prompt,
            'stream':False
        }

        try:
            response=requests.post(self.api_url,json=payload,timeout=10)

            if response.status_code==200:
                return response.json().get('response','No response received.')
            else:
                return f"Error:{response.status_code}-{response.text}"
        except requests.exceptions.RequestException as e:
            return f"Request failed:{e}"
        
if __name__ == "__main__":
    pipeline = TextPipeline()
    symptoms = "Patient has a persistent cough and high fever for three days."
    result = pipeline.process_symptoms(symptoms)
    print("Response:", result)