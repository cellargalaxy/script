from funasr import AutoModel
from typing import List, Dict
import json


def format_recognition_result(res: List[Dict]) -> str:
    formatted_output = []

    for result in res:
        sentences = result["sentence_info"]

        formatted_output.append("语音识别结果：\n")
        for sentence in sentences:
            speaker_id = sentence["spk"]
            text = sentence["text"]
            start_time = sentence["start"] / 1000
            end_time = sentence["end"] / 1000

            formatted_sentence = (
                f"说话人 {speaker_id} "
                f"[{start_time:.2f}s - {end_time:.2f}s]: "
                f"{text}"
            )
            formatted_output.append(formatted_sentence)

    return "\n".join(formatted_output)


model = AutoModel(
    model="Whisper-large-v3", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++"
)

res = model.generate(
    input="208253969-7e35fe2a-7541-434a-ae91-8e919540555d.wav",
)

print(format_recognition_result(res))
