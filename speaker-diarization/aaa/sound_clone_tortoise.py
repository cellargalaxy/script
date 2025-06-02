from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import torch
import torchaudio
import util
import os
import json

logger = util.get_logger()


def sound_clone_one(audio_dir, speaker, text, output_dir):
    device_type = util.get_device_type()
    voice_samples, conditioning_latents = load_voice(speaker, extra_voice_dirs=[audio_dir])
    tts = TextToSpeech(device=torch.device(util.get_device_type()))
    preset = 'ultra_fast'  # ultra_fast / fast / standard / high_quality
    if device_type == 'cuda':
        preset = 'high_quality'
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset=preset,
                              )
    output_path = os.path.join(output_dir, f"{speaker}.wav")
    util.mkdir(output_path)
    torchaudio.save(output_path, gen.squeeze(0).cpu(), sample_rate=24000)


def sound_clone(audio_dir, text, output_dir):
    for speaker in os.listdir(audio_dir):
        speaker_path = os.path.join(audio_dir, speaker)
        sound_clone_one(speaker_path, speaker, text, output_dir)


def sound_clone_by_manager(manager):
    logger.info("sound_clone,enter,manager: %s", json.dumps(manager))
    audio_dir = manager.get('audio_class_dir')
    text = manager.get('sound_clone_text')
    output_dir = os.path.join(manager.get('output_dir'), "sound_clone")
    util.delete_path(output_dir)
    sound_clone(audio_dir, text, output_dir)
    manager['sound_clone_dir'] = output_dir
    logger.info("sound_clone,leave,manager: %s", json.dumps(manager))
