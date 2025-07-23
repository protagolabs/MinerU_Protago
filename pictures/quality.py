from openai import OpenAI
import base64
import mimetypes
import os

# Initialize client
client = OpenAI(
    base_url="https://api.netmind.ai/inference-api/openai/v1",
    api_key="098ba1186ed849bca1180a75383075b7",
)

# Paths
local_image_path = "/home/xing/MinerU_Protago/pictures/f_0AibR1dz_page9_0.jpg"
html_extracted = """
<html><body><table> \t<tr> \t\t<td colspan="2" rowspan="1"></td> \t\t<td colspan="3" rowspan="1">全体 女性 中途採用者</td> \t</tr> \t<tr> \t\t<td colspan="1" rowspan="2">社員</td> \t\t<td colspan="1" rowspan="1"></td> \t\t<td colspan="1" rowspan="1">2,623人</td> \t\t<td colspan="1" rowspan="1">1,575人 (60.0%)</td> \t\t<td colspan="1" rowspan="1">2,408人 (91.8%)</td> \t</tr> \t<tr> \t\t<td colspan="1" rowspan="1">管理職</td> \t\t<td colspan="1" rowspan="1">362人</td> \t\t<td colspan="1" rowspan="1">67人 (18.5%)</td> \t\t<td colspan="1" rowspan="1">300人(82.8%)</td> \t</tr> \t<tr> \t\t<td colspan="2" rowspan="1">取締役</td> \t\t<td colspan="1" rowspan="1">9人</td> \t\t<td colspan="1" rowspan="1">2人 (22.2%)</td> \t\t<td colspan="1" rowspan="1">9人 (100.0%)</td> \t</tr> </table></body></html>
"""

# Detect MIME type
mime_type, _ = mimetypes.guess_type(local_image_path)
if mime_type is None:
    raise ValueError(f"Cannot determine MIME type of {local_image_path}")

# Read and encode the image
if not os.path.exists(local_image_path):
    raise FileNotFoundError(f"File {local_image_path} not found.")

with open(local_image_path, "rb") as f:
    image_bytes = f.read()
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')

# Create data URI
data_uri = f"data:{mime_type};base64,{encoded_string}"

# System prompt
system_prompt = """
You are an HTML Table Examiner.
Your task is to judge the quality of an extracted HTML table against the original table image.
Judge the following aspects:
1. Structure Preservation:
   - Cell Count Accuracy: Does the total number of <td> and <th> cells in the HTML match the number of cells visually in the image?
   - Row Count Accuracy: Does the number of <tr> elements in the HTML equal the number of visible rows (including header rows and data rows) in the image?
   - Column Count Accuracy: Does the HTML preserve the correct number of columns as visually in the image? Check if any columns are missing or extra.
   - Line break preservation: If a single cell in the image contains multi-line text, does the HTML include appropriate line breaks 
   - Row/Column Span Preservation:Does the HTML correctly use rowspan and colspan attributes to represent merged cells? Are multi-row or multi-column headers correctly encoded?
2. Content Accuracy:
   - Exact Cell Content Matches: Do all the cell texts in the HTML exactly match the text in the image (including characters, punctuation, spacing)?
   - Character-level Accuracy for Non-exact Matches: If not exact, how many characters differ? 

Provide a clear evaluation with scores (e.g., 0–10) and comments for each criterion. Please sum up the scores for each criterion and provide the final score.
"""

# Prepare messages
messages = [
    {
        "role": "system",
        "content": system_prompt.strip()
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""Here is the extracted HTML table:
                {html_extracted}
                Please evaluate it against the following table image.
                """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_uri
                }
            }
        ]
    }
]

# Send request
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    messages=messages,
    max_tokens=1024
)

print(response.choices[0].message.content)
