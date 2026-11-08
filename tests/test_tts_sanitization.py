"""Tests for TTS text sanitization, emoji removal, emotion modulation, and pause handling."""

from nino.voice.tts.piper_adapter import (
    clean_text_for_speech,
    extract_actions_and_emotions,
    segment_with_pauses,
    PiperTTSAdapter,
)


def test_clean_text_for_speech_strips_emojis():
    raw = "Hey there! 😊 Glad to see you 🔥! Let's go 🚀"
    cleaned = clean_text_for_speech(raw)
    assert "😊" not in cleaned
    assert "🔥" not in cleaned
    assert "🚀" not in cleaned
    assert cleaned == "Hey there! Glad to see you! Let's go"


def test_clean_text_for_speech_strips_markdown_and_code():
    raw = """# Header
[emotion:happy] **The Cool One**: I'm talking *super* smooth.
```python
x = 10
```
- Item 1 & Item 2
Check [website](http://example.com) for details."""
    cleaned = clean_text_for_speech(raw)
    assert "[emotion:happy]" not in cleaned
    assert "**" not in cleaned
    assert "```" not in cleaned
    assert "x = 10" not in cleaned
    assert "http://example.com" not in cleaned
    assert "website" in cleaned
    assert "and" in cleaned


def test_clean_text_for_speech_converts_special_symbols():
    raw = "Accuracy is 98% and price is $45. Answer is yes/no?"
    cleaned = clean_text_for_speech(raw)
    assert "98 percent" in cleaned
    assert "45 dollars" in cleaned
    assert "yes or no" in cleaned


def test_extract_actions_without_saying_asterisk_content():
    # *chuckles* should be detected as playful and NOT spoken as a word
    raw = "*chuckles* Well, you asked for it."
    processed, emotion, had_action = extract_actions_and_emotions(raw)
    assert "chuckle" not in processed.lower()
    assert emotion == "playful"
    assert had_action is True
    assert "Well, you asked for it." in processed

    # *sighs playfully* should be detected as sarcastic/playful and NOT spoken
    raw2 = "*sighs* Honestly, whatever."
    processed2, emotion2, had_action2 = extract_actions_and_emotions(raw2)
    assert "sigh" not in processed2.lower()
    assert emotion2 == "sarcastic"
    assert had_action2 is True

    # Multi-word roleplay actions should never be spoken
    raw_multi = "*Twirls my avatar, a playful smirk on my face* 'My World' is about you."
    p_multi, emo_multi, had_multi = extract_actions_and_emotions(raw_multi)
    assert "twirls" not in p_multi.lower()
    assert "avatar" not in p_multi.lower()
    assert "smirk" not in p_multi.lower()
    assert had_multi is True
    assert "'My World' is about you." in p_multi

    # Stage directions like *A mischievous glint in my digital eye* should not be spoken
    raw_glint = "*A mischievous glint in my digital eye* Ready?"
    p_glint, emo_glint, had_glint = extract_actions_and_emotions(raw_glint)
    assert "glint" not in p_glint.lower()
    assert "eye" not in p_glint.lower()
    assert had_glint is True
    assert "Ready?" in p_glint

    # *super* is word emphasis -> keep the word!
    raw3 = "I am *super* ready."
    processed3, emotion3, had_action3 = extract_actions_and_emotions(raw3)
    assert "super" in processed3
    assert "*" not in processed3


def test_clean_text_for_speech_strips_shortcodes_and_emoticons():
    raw = "Let's go :rocket: and :fire: :) :D <3 XD"
    cleaned = clean_text_for_speech(raw)
    assert ":rocket:" not in cleaned
    assert ":fire:" not in cleaned
    assert ":)" not in cleaned
    assert ":D" not in cleaned
    assert "<3" not in cleaned
    assert "XD" not in cleaned
    assert cleaned == "Let's go and"



def test_segment_with_pauses():
    text = "Well, you asked for it. Are you ready? Let us see... Yes, sure!"
    segments = segment_with_pauses(text)
    assert len(segments) >= 5

    # Check that commas get clause pause (~200ms)
    comma_seg = [s for s in segments if s[0].endswith(",")][0]
    assert comma_seg[1] == 200

    # Check that period gets sentence pause (~400ms)
    period_seg = [s for s in segments if s[0].endswith(".")][0]
    assert period_seg[1] == 400

    # Check that question mark gets question pause (~450ms)
    question_seg = [s for s in segments if s[0].endswith("?")][0]
    assert question_seg[1] == 450

    # Check that ellipsis gets dramatic pause (~550ms)
    ellipsis_seg = [s for s in segments if s[0].endswith("...")][0]
    assert ellipsis_seg[1] == 550


def test_piper_adapter_cleans_and_pauses_before_synthesis():
    adapter = PiperTTSAdapter(voice="en_US-amy-medium")
    raw = "*chuckles* Testing 😊 **voice** output & clarity, are you ready?"
    audio = adapter._synthesize_sync(raw)
    assert isinstance(audio, bytes)
    assert len(audio) > 0


def test_clean_text_preserves_contractions_and_dashes():
    raw = "I’m pretty sure we shouldn’t do that — it’s just not right… you know?"
    cleaned = clean_text_for_speech(raw)
    assert "I'm" in cleaned
    assert "shouldn't" in cleaned
    assert "it's" in cleaned
    assert "..." in cleaned
    assert ", " in cleaned


def test_piper_adapter_generates_organic_breath_pause():
    adapter = PiperTTSAdapter(voice="en_US-amy-medium")
    normal_breath = adapter._make_breath_pause(ms=450)
    question_breath = adapter._make_breath_pause(ms=450, is_question=True)
    thoughtful_breath = adapter._make_breath_pause(ms=450, is_thoughtful=True)

    assert isinstance(normal_breath, bytes)
    assert len(normal_breath) > 0
    # Questions and thoughtful pauses should adaptively produce longer breath pauses
    assert len(question_breath) > len(normal_breath)
    assert len(thoughtful_breath) > len(question_breath)


def test_insert_breather_pauses_every_4_to_5_words():
    from nino.voice.tts import insert_breather_pauses

    # Long sentence without any punctuation should have breath pauses inserted every 4-5 words
    long_text = "I went to the store and bought fresh fruits vegetables and bread for tomorrow morning."
    paced = insert_breather_pauses(long_text)
    assert "," in paced
    segments = [s.strip() for s in paced.split(",") if s.strip()]
    for seg in segments:
        words = seg.split()
        assert len(words) <= 7

    # Short sentence should remain clean without awkward cuts
    short_text = "Good morning to you."
    assert insert_breather_pauses(short_text) == short_text

    # Existing punctuated sentences should respect existing pauses
    punctuated = "Yes, absolutely, that sounds great."
    assert insert_breather_pauses(punctuated) == punctuated


def test_headline_and_topic_heading_pauses():
    from nino.voice.tts import prepare_headings_with_pauses, HEADLINE_PAUSE_TOKEN

    raw = """Here is the overview for today.

### Key Capabilities
Nino provides persistent memory and on-device tools.

- **Privacy Architecture:** No external data transmission is allowed.

### Next Steps
We can start testing immediately."""

    # 1. Verify token insertion between normal text <-> headings
    prepared = prepare_headings_with_pauses(raw)
    assert HEADLINE_PAUSE_TOKEN in prepared

    sections = [s.strip() for s in prepared.split(HEADLINE_PAUSE_TOKEN.strip()) if s.strip()]
    assert len(sections) == 7
    assert sections[0] == "Here is the overview for today."
    assert sections[1] == "Key Capabilities."
    assert sections[2] == "Nino provides persistent memory and on-device tools."
    assert sections[3] == "Privacy Architecture."
    assert sections[4] == "No external data transmission is allowed."
    assert sections[5] == "Next Steps."
    assert sections[6] == "We can start testing immediately."

    # 2. Verify clean text contains ellipsis pauses before and after headings
    cleaned = clean_text_for_speech(raw)
    assert "..." in cleaned
    # Ensure heading does not bleed directly into following text without pause
    assert "Key Capabilities Nino" not in cleaned
    assert "Next Steps We" not in cleaned

    # 3. Verify Piper audio synthesis inserts breath pause between transitions
    adapter = PiperTTSAdapter(voice="en_US-amy-medium")
    pcm = adapter._synthesize_sync(raw)
    assert isinstance(pcm, bytes)
    assert len(pcm) > 0

