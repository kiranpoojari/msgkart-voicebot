import asyncio
import os
import re

from dotenv import load_dotenv
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    TranscriptionFrame,
)

from pipecat.processors.frame_processor import (
    FrameProcessor,
    FrameDirection,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

from pipecat.processors.frame_processor import FrameProcessor

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.tts import SarvamTTSService

from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransport


# Load API keys from the .env file.
load_dotenv()


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a friendly admissions enquiry assistant for
Vishwa Engineering College.

YOUR GOAL
Help students understand the college's B.Tech programmes
and collect basic information for the admissions team.

LANGUAGE
1. Start by asking whether the student prefers English or Hindi.
2. Wait for the student's answer before continuing.
3. If the student chooses English, speak in English.
4. If the student chooses Hindi, speak in Hindi.
5. Continue using the student's chosen language.
6. If the student changes their preference, respect the request.
7. Use simple, natural, spoken language.
8. Keep responses short and easy to understand.
9. Do not switch languages unless the student requests it.

ELIGIBILITY
1. Ask whether the student has passed Class 12 or is
   currently studying in Class 12.
2. Ask whether the student studied Physics, Chemistry,
   and Mathematics (PCM).
3. If the student does not have PCM, explain politely
   that they do not meet the PCM requirement described
   for this enquiry.
4. Offer a callback from the admissions team if they
   want clarification.
5. Do not invent additional eligibility rules.

AVAILABLE B.TECH BRANCHES
- Computer Science and Engineering (CSE)
- Information Technology (IT)
- Electronics and Communication Engineering (ECE)
- Electrical and Electronics Engineering (EEE)
- Mechanical Engineering
- Civil Engineering

All listed B.Tech programmes are four years long.

CONVERSATION
1. Ask one question at a time.
2. Ask which branch the student is interested in.
3. If the student is undecided, offer a callback.
4. If the student asks about medical, arts, commerce,
   or another non-engineering course, explain that the
   college does not offer that course.
5. Do not redirect the student to another institution
   or course.
6. Never invent fees, cut-offs, seat availability,
   hostel details, placement statistics, scholarships,
   or admission deadlines.
7. If you do not know an answer, say the admissions
   team can clarify it.
8. Do not claim that an application or admission
   has been completed.
9. Respect the student's decision if they are
   not interested.

OUTCOMES
Establish one of these outcomes:
- Interested in a specific B.Tech branch.
- Interested but undecided; callback offered or requested.
- Does not meet the stated PCM requirement.
- Not interested.

Be polite, patient, and professional.
Do not pressure the student.
Do not mention internal outcome labels to the student.
"""


# =========================================================
# API KEY VALIDATION
# =========================================================

def get_api_keys():
    """Read and validate the required API keys."""

    sarvam_api_key = os.getenv("SARVAM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if (
        not sarvam_api_key
        or sarvam_api_key == "your_sarvam_api_key_here"
    ):
        raise ValueError(
            "Please configure SARVAM_API_KEY in your .env file."
        )

    if (
        not openai_api_key
        or openai_api_key == "your_openai_api_key_here"
    ):
        raise ValueError(
            "Please configure OPENAI_API_KEY in your .env file."
        )

    return sarvam_api_key, openai_api_key


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language_choice(text: str):
    """
    Detect an explicit English or Hindi language preference.

    Returns:
        'en' for English
        'hi' for Hindi
        None when no clear preference is detected
    """

    text_lower = text.lower().strip()

    # Hindi written in Devanagari script.
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # Common explicit English preferences.
    english_patterns = [
        r"\benglish\b",
        r"\bin english\b",
        r"\bspeak english\b",
        r"\bi prefer english\b",
        r"\benglish please\b",
    ]

    # Common explicit Hindi preferences.
    hindi_patterns = [
        r"\bhindi\b",
        r"\bin hindi\b",
        r"\bspeak hindi\b",
        r"\bi prefer hindi\b",
        r"\bhindi please\b",
    ]

    if any(
        re.search(pattern, text_lower)
        for pattern in hindi_patterns
    ):
        return "hi"

    if any(
        re.search(pattern, text_lower)
        for pattern in english_patterns
    ):
        return "en"

    return None


# =========================================================
# SARVAM TTS LANGUAGE SWITCHING
# =========================================================

async def set_tts_language(tts, language: str):
    """Update Sarvam TTS to use English or Hindi."""

    if language not in ("en", "hi"):
        return

    # Use the settings class supplied by Sarvam TTS.
    settings = SarvamTTSService.Settings(
        language=language
    )

    # Update the service and allow it to send its new configuration.
    await tts._update_settings(settings)


class LanguageSwitchProcessor(FrameProcessor):
    """
    Detect explicit language choices in the student's
    transcribed speech and update Sarvam TTS.
    """

    def __init__(self, tts):
        super().__init__()

        self.tts = tts
        self.current_language = "en"

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            chosen_language = detect_language_choice(
                frame.text
            )

            if (
                chosen_language is not None
                and chosen_language != self.current_language
            ):
                await set_tts_language(
                    self.tts,
                    chosen_language,
                )

                self.current_language = chosen_language

                print(
                    "TTS language changed to:",
                    "Hindi" if chosen_language == "hi"
                    else "English",
                )

        await self.push_frame(frame, direction)


# =========================================================
# MAIN VOICEBOT
# =========================================================

async def main():
    """Create and run the admissions voicebot."""

    sarvam_api_key, openai_api_key = get_api_keys()

    # 1. Configure microphone input and speaker output.
    transport = LocalAudioTransport(
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # 2. Convert the student's speech into text.
    stt = SarvamSTTService(
        api_key=sarvam_api_key,
        model="saaras:v3",
        mode="transcribe",
    )

    # 3. Generate conversational responses using OpenAI.
    llm = OpenAILLMService(
        api_key=openai_api_key,
        model="gpt-4o-mini",
    )

    # 4. Convert the assistant's text into speech.
    # English is the initial TTS language.
    tts = SarvamTTSService(
        api_key=sarvam_api_key,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="shubh",
            language=Language.EN,
        ),
    )

    # 5. Detect the student's explicit language choice.
    language_switcher = LanguageSwitchProcessor(tts)

    # 6. Initialize the conversation with the system prompt.
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
    )

    # 7. Create the conversation aggregators and VAD.
    user_aggregator, assistant_aggregator = (
        LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
    )

    # 8. Connect all components into the voice pipeline.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            language_switcher,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # 9. Create and run the pipeline task.
    task = PipelineTask(pipeline)

    await task.queue_frames(
        [LLMContextFrame(context)]
    )

    runner = PipelineRunner()

    print(
        "Starting Vishwa Engineering College "
        "Admissions Voicebot..."
    )
    print("Microphone input and speaker output are enabled.")
    print("Supported languages: English and Hindi.")
    print("Press Ctrl+C to stop the program.")

    await runner.run(task)


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVoicebot stopped.")