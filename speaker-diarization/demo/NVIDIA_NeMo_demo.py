# 1. 安装或更新NeMo
# !pip install nemo_toolkit['asr']

import nemo.collections.asr as nemo_asr
import json
import os
from omegaconf import OmegaConf # NeMo使用OmegaConf来管理复杂的配置

# 2. 加载预训练的MSDD Diarization模型
#    这个模型内部集成了VAD、声纹提取(TitaNet)和说话人变化检测(MSDD)
print("Loading Diarization Model...")
diarization_model = nemo_asr.models.ClusterDiarizer.from_pretrained(
    model_name="diar_msdd_telephonic_wpe_titanet_ls" 
)
print("Model Loaded.")

# 3. 准备manifest文件（不设置num_speakers）
# <<< 关键改动：将num_speakers设为None或移除
audio_file_path = "../demo_eng_single.wav" # 替换成你的文件路径

meta = {
    'audio_filepath': audio_file_path,
    'offset': 0,
    'duration': None,
    'label': 'infer',
    'text': '-',
    'num_speakers': None, # <<< 明确设为None，告诉NeMo去自动检测
    'rttm_filepath': None,
    'uem_filepath': None
}

manifest_path = 'input_manifest.json'
with open(manifest_path, 'w') as f:
    f.write(json.dumps(meta) + '\n')

# 4. 执行分割
#    当num_speakers为None时，NeMo的diarize方法会自动调用MSDD流程
print("Starting Diarization...")
output_dir = './nemo_output_unknown_speakers'
os.makedirs(output_dir, exist_ok=True)

diarization_model.diarize(
    paths2audio_files=[manifest_path],
    output_dir=output_dir
)
print(f"Diarization complete. Results saved in: {output_dir}")

# 结果文件位于: nemo_output_unknown_speakers/pred_rttms/audio_16k.rttm
# 你可以读取这个RTTM文件来查看分割结果
rttm_file = os.path.join(output_dir, 'pred_rttms', os.path.basename(audio_file_path).replace('.wav', '.rttm'))
if os.path.exists(rttm_file):
    print("\n--- RTTM Result ---")
    with open(rttm_file, 'r') as f:
        print(f.read())
else:
    print(f"Could not find RTTM file at {rttm_file}")