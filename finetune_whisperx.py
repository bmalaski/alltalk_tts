import whisperx
from faster_whisper.transcribe import Word

class WhisperXTranscribe:
    def __init__(self, whisper_model, target_language, device, whisperx_alignment):
        self.whisper_model = whisper_model
        self.device = device
        self.whisperx_alignment = whisperx_alignment
        self.target_language = target_language
        self.whisperx_model = whisperx.load_model(self.whisper_model, language=self.target_language, device=self.device, compute_type="float32")
        if (self.whisperx_alignment):
            self.model_a, self.metadatax = whisperx.load_align_model(language_code=self.target_language, device=self.device)

    def deleteWhisperX(self):
        if(self.whisperx_model):
            del self.whisperx_model
        if(self.model_a is not None):
            del self.model_a

    def transcribeWhisperX(self, audio_path, whisperx_batch_size, wav, words_list):
        print("Start transcribing...")

        # audio = whisperx.load_audio(audio_path)
        result = self.whisperx_model.transcribe(audio_path, language=self.target_language, batch_size=whisperx_batch_size)
        print(result["segments"])
        if (self.whisperx_alignment):
            print(f" [FINETUNE] Starting Alignment")
            result = whisperx.align(result["segments"], self.model_a, self.metadatax, wav, self.device, return_char_alignments=False)
            print(result["segments"])
        segments = result["segments"]
        for segment in enumerate(segments):
            for word in segment[1]['words']:
                word_instance = Word(start=word['start'], end=word['end'], word=word['word'],
                                     probability=word['score'])
                words_list.append(word_instance)

        print("Done transcribing...")



