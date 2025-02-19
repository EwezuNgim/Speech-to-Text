import streamlit, soundfile as sf, transformers, num2words 
import streamlit as st
import tempfile
import torch, os


st.title("Text-to-Speech (TTS) App")

# Text Input
text = st.text_area("Enter text for me to convert to speech:", "Hello, welcome to my TTS app!")

if st.button("Generate Speech"):
    if text.strip():
        # Convert text to speech
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset, load_from_disk
        # First, load the base model
        model = SpeechT5ForTextToSpeech.from_pretrained("toyrem/speecht5_en-ng")
        from text_processing import process_text
        # Load the processor and the vocoder
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        final_text = process_text(text)
        inputs = processor(text=final_text, return_tensors='pt')
        # Load Embedding Dataset
        embeddings_dataset=load_from_disk('cmu_artic_embeddings')
        #embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        #embeddings_dataset.save_to_disk('cmu_artic_embeddings')
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
        gain = 4.0
        louder_audio = speech.numpy() * gain
        # Clip audio to prevent distortion
        louder_audio = louder_audio.clip(-1.0, 1.0)
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name,louder_audio,samplerate=16000)
            temp_audio_path = temp_audio.name

        # Display Audio
        st.audio(temp_audio_path, format="wav")

    else:
        st.warning("Please enter some text to generate speech.")
