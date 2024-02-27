"""
Script used to prerender a Wikipedia dump that has been downloaded to disk.
Processes the dataset line-by-line, extracting documents (articles) and uploads the rendered examples in chunks
to HuggingFace. Tries to filter out title lines as these are typically short and provide little value.
Examples are stored and compressed in parquet files.
Relies on a modified version of the datasets library installed through git submodule.

export RENDERER_PATH="configs/renderers/noto_renderer"
python scripts/data/prerendering/prerender_wikipedia_multilingual.py \
  --renderer_name_or_path="configs/renderers/noto_renderer" \
  --chunk_size=100000 \
  --repo_id="stefania-radu/PIXEL-semantic" \
  --split="train" \
  --auth_token="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"

languages: [am, ha, ig, rw, lg, pcm, sw, wo, yo, ar, fi, id, ru, te]
"""

import argparse
import logging
import sys
from datasets import load_dataset
from PIL import Image
from gi.repository import PangoCairo
from pixel import PangoCairoTextRenderer, log_example_while_rendering, push_rendered_chunk_to_hub

logger = logging.getLogger(__name__)

def process_doc(
    args: argparse.Namespace,
    text_renderer: PangoCairoTextRenderer,
    idx: int,
    data: dict,
    dataset_stats: dict,
    doc: str,
    target_seq_length: int,
):
    doc = doc.strip().split("\n")

    width = 0
    block = []
    for line in doc:

        dataset_stats["total_num_words"] += len(line.split(" "))


        # Create a temporary surface and context
        surface, context, sep_patches = text_renderer.get_empty_surface()

        # Create a Pango Layout
        layout = PangoCairo.create_layout(context)
        layout.set_font_description(text_renderer.font)
        layout.set_text(line, -1)

        # Measure the width
        line_width, _ = layout.get_pixel_size()

        # line_width = text_renderer.font.get_rect(line).width

        
        if width + line_width >= target_seq_length:
            idx += 1
            sequence = " ".join(block)

            encoding = text_renderer(text=sequence)
            data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
            data["num_patches"].append(encoding.num_text_patches)

            if idx % args.chunk_size == 0:
                log_example_while_rendering(idx, sequence, encoding.num_text_patches)
                dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
                data = {"pixel_values": [], "num_patches": []}

            width = line_width
            block = [line]
        else:
            block.append(line)
            width += line_width

    if len(block) > 0:
        idx += 1
        sequence = " ".join(block)
        encoding = text_renderer(text=sequence)

        data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
        data["num_patches"].append(encoding.num_text_patches)

        if idx % args.chunk_size == 0:
            print("idx mod args.chunk_size == 0")
            log_example_while_rendering(idx, sequence, encoding.num_text_patches)
            dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
            data = {"pixel_values": [], "num_patches": []}

    return idx, data, dataset_stats


def main(args: argparse.Namespace):
    # Load PyGame renderer
    text_renderer = PangoCairoTextRenderer.from_pretrained(args.renderer_name_or_path, use_auth_token=args.auth_token)

    data = {"pixel_values": [], "num_patches": []}
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    max_pixels = text_renderer.pixels_per_patch * text_renderer.max_seq_length - 2 * text_renderer.pixels_per_patch
    target_seq_length = max_pixels

    idx = 0
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{args.lang}", streaming=True)

    print(ds['train'])

    for document in ds['train']:
        text = document['text']
        # Process the document text
        idx, data, dataset_stats = process_doc(
            args=args,
            text_renderer=text_renderer,
            idx=idx,
            data=data,
            dataset_stats=dataset_stats,
            doc=text,
            target_seq_length=target_seq_length,
        )

    # Push final chunk to hub
    push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
    logger.info(f"Total num words in wikipedia: {dataset_stats['total_num_words']}")


if __name__ == "__main__":

    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer_name_or_path",
        type=str,
        help="Path or Huggingface identifier of the text renderer",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to a dataset on disk")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Push data to hub in chunks of N lines",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=-1,
        help="Only look at the first N non-empty lines",
    )
    parser.add_argument("--repo_id", type=str, help="Name of dataset to upload")
    parser.add_argument("--split", type=str, help="Name of dataset split to upload")
    parser.add_argument("--lang", type=str, help="The language of the wikipedia to be rendered. here: [am, ha, ig, rw, lg, pcm, sw, wo, yo, ar, fi, id, ru, te] ")
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Huggingface auth token with write access to the repo id",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
