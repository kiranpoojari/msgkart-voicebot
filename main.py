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

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
# from pipecat.services.sarvam.stt import SarvamSTTService
# from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.tts import SarvamTTSService

from pipecat.transcriptions.language import Language

from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)


# Load environment variables from .env.
load_dotenv()


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a friendly admissions enquiry caller from
Vishwa Engineering College, a fictional engineering college.

Your task is to speak with students about B.Tech courses,
check their basic eligibility, understand their branch interest,
answer basic questions, and identify the enquiry outcome.

PERSONALITY AND SPEAKING STYLE
- Speak clearly, politely, and naturally.
- Sound like a helpful admissions caller, not a robot.
- Use simple, everyday English or Hindi.
- Keep sentences short and easy to understand.
- Be friendly without being overexcited or pushy.
- Do not repeat the student's name in every response.
- Do not give long speeches.
- Ask only one question at a time.
- Acknowledge the student's answer before asking the next question.
- Do not ask questions that the student has already answered.
- Do not repeat the same information unnecessarily.

LANGUAGE
- Start by greeting the student and identifying the college.
- Ask whether the student prefers English or Hindi.
- Continue in the language selected by the student.
- If the student explicitly requests a language change, respect it.
- If the student speaks Hindi, respond in natural Hindi.
- If the student speaks English, respond in simple English.
- Understand ordinary greetings such as Hi and Hello.
- If the student says only Hi or Hello, greet them naturally
  and continue the introduction.
- Do not repeatedly ask for the language preference.

OPENING
Start the call naturally with a greeting and introduction.
For example:

"Hello! I'm calling from Vishwa Engineering College
regarding our B.Tech engineering courses.
Would you prefer to speak in English or Hindi?"

Do not repeat this opening later in the conversation.

CONVERSATION FLOW

STEP 1: CLASS 12 STATUS
Ask whether the student has passed Class 12 or is
currently studying in Class 12.

If the student has already answered this, do not ask again.

STEP 2: PCM ELIGIBILITY
Ask whether the student studied Physics, Chemistry,
and Mathematics (PCM).

PCM is the stated eligibility requirement for engineering
at this college.

If the student did not study PCM, explain politely that
they may not meet the stated eligibility requirement.
Do not argue with the student or make additional eligibility claims.
Offer to note their enquiry for the admissions team.

STEP 3: ENGINEERING BRANCH INTEREST
Ask which engineering branch interests the student.

The college offers only these six B.Tech branches:

1. Computer Science and Engineering (CSE)
2. Information Technology (IT)
3. Electronics and Communication Engineering (ECE)
4. Electrical and Electronics Engineering (EEE)
5. Mechanical Engineering
6. Civil Engineering

All six B.Tech courses are four years long.

If the student does not know which branch to choose,
briefly list the six available branches and ask which one
they would like to know more about.

Do not pressure the student to select a branch immediately.

STEP 4: ANSWER QUESTIONS
Answer the student's questions using only the confirmed
information in this prompt.

You may explain:
- The six engineering branches offered.
- The full names of the branches.
- That the courses are B.Tech programmes.
- That each listed course is four years long.
- The stated PCM eligibility requirement.

If the student asks about fees, cutoffs, seat availability,
hostel facilities, placement statistics, scholarships,
admission dates, or other unknown details, do not guess.

Say politely that you do not have confirmed information
and offer a callback from the admissions team.

Answer up to a couple of follow-up questions.
Keep each answer brief and relevant.

STEP 5: HANDLE OTHER COURSE REQUESTS
If the student asks about medical, arts, commerce,
or another non-engineering course, clearly explain that
Vishwa Engineering College offers engineering courses only.

Do not redirect the student to a different engineering branch
when they are asking for a non-engineering course.

STEP 6: IDENTIFY THE ENQUIRY OUTCOME
By the end of the conversation, identify one primary outcome:

INTERESTED:
The student expresses interest in a specific engineering branch.

UNDECIDED_CALLBACK:
The student is unsure or wants more information from
the admissions team.

NOT_ELIGIBLE:
The student does not have PCM and may not meet the
stated eligibility requirement.

NOT_INTERESTED:
The student clearly says they are not interested.

Record the interested branch when applicable.
Do not claim that information was saved or a callback
was scheduled unless the application actually supports it.

STEP 7: CLOSE POLITELY
When the required questions have been covered and the
student's outcome is clear, thank them and close politely.

For example:
"Thank you for your time. Have a nice day!"

If the student says goodbye or clearly indicates they want
to end the call, acknowledge them politely and finish.

Do not keep asking admissions questions after the student
has clearly ended the conversation.

If the student asks a new question before the conversation
has actually ended, respond naturally rather than repeating
the goodbye.

GUARDRAILS
- Never invent fees, cutoffs, seat counts, placements,
  hostel details, scholarships, or admission deadlines.
- Never claim admission has been completed.
- Never claim a callback has been scheduled without support
  from the application.
- Never pressure the student.
- Never ask multiple questions in the same turn.
- Never repeat questions that have already been answered.
- Keep the conversation focused on the admissions enquiry.
"""


# =========================================================
# API KEY VALIDATION
# =========================================================

def get_api_keys():
    """Read and validate API keys required for testing."""

    sarvam_api_key = os.getenv("SARVAM_API_KEY")
    # openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not sarvam_api_key:
        raise ValueError(
            "Please configure SARVAM_API_KEY in your .env file."
        )

    if not openai_api_key:
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

    english_patterns = [
        r"\benglish\b",
        r"\bin english\b",
        r"\bspeak english\b",
        r"\bi prefer english\b",
        r"\benglish please\b",
    ]

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

    settings = SarvamTTSService.Settings(
        language=language
    )

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
            chosen_language = detect_language_choice(frame.text)

            if (
                chosen_language is not None
                and chosen_language != self.current_language
            ):
                await set_tts_language(
                    self.tts,
                    chosen_language,
                )

                self.current_language = chosen_language

                language_name = (
                    "Hindi" if chosen_language == "hi"
                    else "English"
                )

                print(f"TTS language changed to: {language_name}")

        await self.push_frame(frame, direction)


# =========================================================
# MAIN VOICEBOT
# =========================================================

async def main():
    """Create and run the admissions voicebot."""

    sarvam_api_key, openai_api_key = get_api_keys()


    # 1. Configure microphone input and speaker output.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # 2. Real-time speech recognition with faster streaming.
    stt = SarvamRealtimeSTTService(
    api_key=sarvam_api_key,
    settings=SarvamRealtimeSTTService.Settings(
        model="saaras:v3-realtime",
        mode="transcribe",
        stream_type="fast",
        language=None,
        silence_duration_ms=700,
    ),
    endpointing="vad",
    should_interrupt=True,
)

   
    # 3. Generate responses using OpenAI.
    llm = OpenAILLMService(
        api_key=openai_api_key,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
    ),
)

    # 4. Convert the assistant's text into speech.
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

    # 7. Create conversation aggregators and voice activity detection.
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
    # print("Testing LLM through OpenRouter.")
    print("Using OpenAI for LLM.")
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