# MsgKart Voicebot Orchestrator

## About

This project is a real-time voicebot built using Python and Pipecat.

It acts as an admissions enquiry assistant for Vishwa Engineering College and supports conversations in English and Hindi.

## Technologies Used

- Python
- Pipecat
- Sarvam AI for speech-to-text and text-to-speech
- OpenAI API for generating responses

## How It Works

1. The user speaks through a microphone.
2. Sarvam AI converts speech into text.
3. OpenAI generates a response based on the system prompt.
4. Sarvam AI converts the response into speech.
5. The response plays through the speaker.

The assistant asks about Class 12 status, PCM eligibility, and the student's preferred engineering branch. It avoids inventing admission information.

## Requirements

- Python 3.12
- Sarvam AI API key
- OpenAI API key with available credits
- Microphone and speaker
- Internet connection

## Setup

Clone the repository:

```bash
git clone https://github.com/kiranpoojari/msgkart-voicebot.git
cd msgkart-voicebot
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder and add your API keys:

```env
SARVAM_API_KEY=your_sarvam_api_key
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder values with your own API keys. Never upload the `.env` file to GitHub.

Run the application:

```bash
python main.py
```

## Testing

The application was run locally. The Sarvam speech-to-text and text-to-speech connections were established.

During testing, the OpenAI API returned an insufficient-quota error because the account had no API credits remaining. Therefore, a complete end-to-end voice conversation could not be verified.

## Note

The application requires valid API keys, available API credits, and an active internet connection.