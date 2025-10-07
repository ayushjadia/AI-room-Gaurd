import os
import json
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
load_dotenv()

client = Cerebras(
    # This is the default and can be omitted
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

# movie_schema = {
#     "type": "object",
#     "properties": {
#         "title": {"type": "string"},
#         "director": {"type": "string"},
#         "year": {"type": "integer"},
#     },
#     "required": ["title", "director", "year"],
#     "additionalProperties": False
# }

# completion = client.chat.completions.create(
#     model="llama3.1-8b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant that generates movie recommendations."},
#         {"role": "user", "content": "Suggest a sci-fi movie from the 1990s"}
#     ],
#     response_format={
#         "type": "json_schema", 
#         "json_schema": {
#             "name": "movie_schema",
#             "strict": True,
#             "schema": movie_schema
#         }
#     }
# )

# # Parse the JSON response
# movie_data = json.loads(completion.choices[0].message.content)
# # print(json.dumps(movie_data, indent=2))

stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "My name is Ayush"
        }
    ],
    model="llama3.1-8b",
    stream=True,
    max_completion_tokens=20000,
    temperature=0.7,
    top_p=0.8
)
# for chunk in stream:
#   print(chunk.choices[0].delta.content or "", end="")