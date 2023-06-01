import os
import torch
import torchaudio
import sox
import json
import argparse


from text_normalization import text_normalize
from align_utils import (
    get_uroman_tokens,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
)
import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)

    audio_sf = sox.file_info.sample_rate(audio_file)
    waveform = F.resample(waveform, orig_freq=audio_sf, new_freq=SAMPLING_FREQ)
    # assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, stride


def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()


    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]
    
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    input_lengths = torch.tensor(emissions.shape[0])
    target_lengths = torch.tensor(targets.shape[0])

    path, _ = F.forced_align(
        emissions, targets, input_lengths, target_lengths, blank=blank
    )
    path = path.to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride


def main(outdir, text_filepath, lang, uroman_path, audio_filepath):
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device: ", DEVICE)
    
    assert not os.path.exists(
        outdir
    ), f"Error: Output path exists already {outdir}"
    print(f"Output will be saved in {outdir}")
    transcripts = []
    with open(text_filepath) as f:
        transcripts = [line.strip() for line in f]
    print("Read {} lines from {}".format(len(transcripts), text_filepath))

    norm_transcripts = [text_normalize(line.strip(), lang) for line in transcripts]
    tokens = get_uroman_tokens(norm_transcripts, uroman_path, lang)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)


    segments, stride = get_alignments(
        audio_filepath,
        tokens,
        model,
        dictionary    
    )
    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)

    os.makedirs(outdir)
    with open( f"{outdir}/manifest.json", "w") as f:
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end

            output_file = f"{outdir}/segment{i}.flac"

            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000 

            tfm = sox.Transformer()
            tfm.trim(audio_start_sec , audio_end_sec)
            tfm.build_file(audio_filepath, output_file)
            
            sample = {
                "audio_start_sec": audio_start_sec,
                "audio_filepath": str(output_file),
                "duration": audio_end_sec - audio_start_sec,
                "text": t,
                "normalized_text":norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            f.write(json.dumps(sample) + "\n")

