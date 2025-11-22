import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import util

logger = util.get_logger()


def exec(manager):
    import split_audio

    import init
    init.exec(manager)

    import extract_audio
    extract_audio.exec(manager)

    import extract_stem
    import extract_stem_uvr
    extract_stem.exec(manager, [
        extract_stem_uvr.VocalHandler(),
        extract_stem_uvr.MainVocalHandler(),
        extract_stem_uvr.DeReverbHandler('MDX23C-De-Reverb-aufr33-jarredou.ckpt'),
    ])
    return

    import extract_loudness
    extract_loudness.exec(manager)

    import extract_simple
    extract_simple.exec(manager)

    import part_detect
    part_detect.exec(manager)
    # split_audio.exec(manager, 'part_detect_path')

    import part_divide
    part_divide.exec(manager)
    # split_audio.exec(manager, 'part_divide_path')

    import segment_detect
    segment_detect.exec(manager)
    # split_audio.exec(manager, 'segment_detect_path')

    import segment_divide
    segment_divide.exec(manager)
    # split_audio.exec(manager, 'segment_divide_path')

    import speaker_detect
    speaker_detect.exec(manager)
    split_audio.exec(manager, 'speaker_detect_path')


def exec_batch(video_paths):
    for i, video_path in enumerate(video_paths):
        try:
            manager = {
                "video_path": video_path,
            }
            exec(manager)
        except Exception as e:
            logger.error("exec_batch failed.", exc_info=True)
            util.input_timeout("异常，回车继续: ", 60)


video_paths = [
    # '../material/panty.mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 01 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 02 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 03 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 04 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 05 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 06 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 07 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 08 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 09 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    # '../material/panty/[Nekomoe kissaten&LoliHouse] New PANTY & STOCKING with GARTERBELT - 10 [WebRip 1080p HEVC-10bit AACx2 ASSx2].mkv',
    '../material/my_way.webm',
    # '../material/mao.mp3',
    # '../material/demo.mkv',
    # '../material/mkv.mkv',
    # '../material/holo.mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」 #1　あの賢狼ホロがYouTuberデビュー⁉ [Vup1WxjRaUY].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」 #2　「写真で一言」でまさかの珍回答⁉ [A1LoxukWcrs].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」 #3 「音あてクイズ」で賢狼の本領発揮⁉ [1LbrvnMbvKY].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#10   賢狼ホロについてなんでも答えます [d0BpnJD0L6o].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#11 ホロがオンラインくじに挑戦!! [MduFhvNCHY8].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#12 ロレンスが破産寸前になった理由を解説 [CNso0F6GKwE].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#13 羊飼いノーラに電話してみた [YZearp10ADk].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#14 第２クール突入‼ 写真でひとこと③ [HRwLMEUDpWg].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#15 ホロがカラオケで遊んでみた [X0sVerXNiTc].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#16 ぬしっ子たちへの感謝を込めて [Ox3Mwzaa7hk].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#17 ロレンス VS アマーティ 全力解説 [6Cdf9rH1sMA].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#18 「狼と100時間」企画結果発表⁉ [wEJtQwTn4-8].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#19 賢狼ホロがズバッと解決!! [68Vh_6uhoj4].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#20 「若き魚商人との決闘」篇の舞台 クメルスン解説 [xmVAi7yeh_Y].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#21 Blu-ray映像特典 配信間近！ASMRクイズに挑戦!! [yzAjxbWZyLo].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#22 ただおたよりを読む回 [sh04E8ynRjo].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#23 これまでの「異教の神々の伝承」篇 [eqLKSpEGawM].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#24 お悩み相談でまさかの〇レンス登場!？ [f93Pf2SMP1I].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#25 ホロからの重大告知 [qSUnQEKv8Zc].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#26 最終回 エルサも登場!？ [27TjuXKg3Lc].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#4 賢狼ホロが現代人の悩みをズバッと解決⁉ [WNyMMxuczRk].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#5 賢狼が自分の精巧フィギュアに赤面連発⁇ [tULw8-tb0J0].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#6 第1章「新たな旅の始まり」篇ふりかえりクイズに挑戦!! [xv40oQhVROw].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#7  ＜写真でひとこと＞でもはや大喜利状態!？ [GdftVxWwKxQ].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#8   賢狼ホロが迷える子羊たちを導きます [yzH8cgGWVJc].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#9   アニメ番宣CMのナレーションに挑戦‼ [2dgxjniwouw].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#Ex2 OPテーマを歌うHana Hopeさんを賢狼ホロがインタビュー!! [O0TLu3HKOuY].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」#Ex3 EDテーマを歌うClariSさんを賢狼ホロがインタビュー!! [2X3WFG7tEhQ].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」Ex.4 第2クールOPテーマを担当するAimerさんをインタビュー!! [7yonSxvkJuw].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」Ex.5 第2クールEDテーマを担当する音莉飴さんと初の＂バーチャルコラボ＂!! [9Kml47ObYbk].webm',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」ホロがナレーションを当てた特別PV① [fxUz3Syfx5Y].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」ホロがナレーションを当てた特別PV② [iMHU4LRbS1o].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」ホロがナレーションを当てた特別PV③ [pH1sdfPOi10].mkv',
    # '../material/youtube-live/【狼と香辛料】「賢狼ホロのわっちチャンネル」レストランHoloコラボ記念 トリプルデラックスグリルによだれがじゅるり…… [WNUTn4vgmnM].webm',
]

exec_batch(video_paths)
