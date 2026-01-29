#!/usr/bin/env python3
"""
Generate a comprehensive prompt set for ICML-style evaluation.

Outputs (by default):
  - data/prompt_icml_comprehensive_1000.txt
  - data/prompt_icml_calib_64.txt

Design goals:
  - Broad coverage (portraits, animals, products, architecture, etc.)
  - Stress test failure modes (text rendering, counting, small details, reflections)
  - Deterministic output given a seed
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _clean(prompt: str) -> str:
    prompt = " ".join(prompt.split())
    return prompt.strip().rstrip(",")


def _pick(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def _maybe(rng: random.Random, p: float, text: str) -> str:
    return text if rng.random() < p else ""


def _unique_add(out: list[str], seen: set[str], prompt: str) -> None:
    prompt = _clean(prompt)
    if not prompt:
        return
    if prompt in seen:
        return
    seen.add(prompt)
    out.append(prompt)


def gen_portraits(rng: random.Random, seen: set[str], count: int) -> list[str]:
    people = [
        "a middle-aged woman with freckles and curly hair",
        "a young man with round glasses and a denim jacket",
        "an elderly man with a salt-and-pepper beard",
        "a teenage girl with braided hair and a nose ring",
        "a woman with short platinum hair and subtle makeup",
        "a man with dark curly hair and a warm smile",
        "a woman wearing a patterned headscarf",
        "a man with a shaved head and a leather jacket",
        "a woman with wavy auburn hair and green eyes",
        "a man with a mustache and a tweed blazer",
        "a person with vitiligo and soft studio lighting",
        "a person with a faint scar on the eyebrow and calm expression",
        "a person with a knit beanie and winter coat",
        "a person with long black hair and natural skin texture",
    ]
    expressions = [
        "soft smile",
        "serious expression",
        "laughing",
        "thoughtful gaze",
        "looking slightly off-camera",
        "direct eye contact",
    ]
    lighting = [
        "soft window light",
        "dramatic rim lighting",
        "golden hour sunlight",
        "overcast daylight",
        "studio softbox lighting",
        "neon-lit night ambience",
        "candlelit warm tones",
    ]
    camera = [
        "85mm portrait lens",
        "50mm lens",
        "shallow depth of field",
        "sharp focus on the eyes",
        "natural film look",
        "high dynamic range",
    ]
    backgrounds = [
        "a blurred city street background",
        "a dark studio background",
        "a softly lit indoor background",
        "a bright minimal background",
        "a cozy cafe interior background",
        "a garden background with bokeh",
    ]
    details = [
        "visible skin pores",
        "fine hair strands",
        "catchlights in the eyes",
        "natural skin tones",
        "high detail facial features",
    ]

    templates = [
        "Photorealistic close-up portrait of {person}, {expr}, {light}, {bg}, {cam}, {detail}.",
        "Half-body portrait of {person}, hands holding {prop}, {expr}, {light}, {bg}, {cam}, {detail}.",
        "Environmental portrait of {person} in {env}, {expr}, {light}, {cam}, {detail}.",
    ]
    props = [
        "a ceramic mug",
        "a paperback book",
        "a small bouquet of flowers",
        "a camera",
        "a glass of iced tea",
        "a notebook and pen",
    ]
    envs = [
        "a quiet library",
        "a photography studio",
        "a modern kitchen",
        "a workshop with tools",
        "a subway station entrance",
        "a seaside promenade",
        "a rainy street under an umbrella",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        t = _pick(rng, templates)
        prompt = t.format(
            person=_pick(rng, people),
            expr=_pick(rng, expressions),
            light=_pick(rng, lighting),
            bg=_pick(rng, backgrounds),
            cam=", ".join(rng.sample(camera, k=2)),
            detail=", ".join(rng.sample(details, k=2)),
            prop=_pick(rng, props),
            env=_pick(rng, envs),
        )
        _unique_add(out, seen, prompt)

    return out


def gen_animals(rng: random.Random, seen: set[str], count: int) -> list[str]:
    animals = [
        ("a red fox", "fur"),
        ("a snowy owl", "feathers"),
        ("a hummingbird", "feathers"),
        ("a golden retriever", "fur"),
        ("a black cat", "fur"),
        ("a leopard", "fur"),
        ("a sea turtle", "scales"),
        ("a koi fish", "scales"),
        ("a chameleon", "scales"),
        ("a peacock", "feathers"),
        ("a horse", "fur"),
        ("a polar bear", "fur"),
        ("a scarlet macaw", "feathers"),
    ]
    actions = [
        "running through tall grass",
        "jumping over snow",
        "taking off in flight",
        "splashing water",
        "sleeping curled up",
        "looking directly at the camera",
        "shaking off rain droplets",
    ]
    habitats = [
        "in a misty forest",
        "on a rocky coastline",
        "in a snowy field",
        "beside a calm lake",
        "in a sunlit meadow",
        "in a dense jungle",
        "in a desert canyon",
    ]
    lighting = [
        "golden hour backlight",
        "soft overcast daylight",
        "dramatic storm light",
        "neon-lit night reflections",
        "bright midday sun with crisp shadows",
    ]
    camera = [
        "telephoto lens",
        "high-speed shutter",
        "sharp focus",
        "shallow depth of field",
        "natural colors",
        "high detail",
    ]
    templates = [
        "Wildlife photograph of {animal} {action} {habitat}, {light}, {cam}, detailed {tex}.",
        "Close-up photo of {animal} {action}, {light}, {cam}, fine {tex} detail.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        animal, tex = rng.choice(animals)
        prompt = _pick(rng, templates).format(
            animal=animal,
            action=_pick(rng, actions),
            habitat=_pick(rng, habitats),
            light=_pick(rng, lighting),
            cam=", ".join(rng.sample(camera, k=3)),
            tex=tex,
        )
        _unique_add(out, seen, prompt)
    return out


def gen_food(rng: random.Random, seen: set[str], count: int) -> list[str]:
    dishes = [
        "a flaky croissant on a wooden board",
        "a bowl of ramen with glossy broth and noodles",
        "a strawberry tart with shiny glaze",
        "sushi on a black slate plate",
        "a cheeseburger with melted cheese and sesame bun",
        "a cappuccino with detailed latte art",
        "a stack of pancakes with syrup and butter",
        "a fresh salad with avocado and tomatoes",
        "a slice of chocolate cake with ganache",
        "dumplings in a bamboo steamer",
        "a grilled steak with seared crust",
        "a plate of colorful macarons",
    ]
    lighting = [
        "soft natural window light",
        "studio food lighting",
        "warm restaurant lighting",
        "overhead diffused light",
    ]
    camera = [
        "macro lens",
        "shallow depth of field",
        "sharp focus",
        "high detail texture",
        "realistic color",
    ]
    angles = [
        "top-down composition",
        "45-degree angle",
        "close-up framing",
    ]
    details = [
        "visible crumbs and flaky layers",
        "condensation droplets",
        "steam rising subtly",
        "crisp highlights on glossy surfaces",
        "fine texture on the plate",
    ]

    templates = [
        "Macro food photograph of {dish}, {angle}, {light}, {cam}, {detail}.",
        "High-resolution food photo of {dish}, {light}, {cam}, {angle}, {detail}.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        prompt = _pick(rng, templates).format(
            dish=_pick(rng, dishes),
            angle=_pick(rng, angles),
            light=_pick(rng, lighting),
            cam=", ".join(rng.sample(camera, k=3)),
            detail=", ".join(rng.sample(details, k=2)),
        )
        _unique_add(out, seen, prompt)
    return out


def gen_products(rng: random.Random, seen: set[str], count: int) -> list[str]:
    products = [
        "a smartwatch",
        "a pair of wireless earbuds",
        "a mechanical keyboard",
        "a camera lens",
        "a perfume bottle",
        "a ceramic vase",
        "a stainless steel water bottle",
        "a pair of running shoes",
        "a fountain pen",
        "a compact flashlight",
        "a vintage wristwatch",
        "a coffee grinder",
    ]
    materials = [
        "brushed aluminum",
        "matte black plastic",
        "clear glass",
        "polished stainless steel",
        "ceramic glaze",
        "carbon fiber",
        "soft-touch rubber",
        "anodized metal",
    ]
    backgrounds = [
        "on a seamless white background",
        "on a matte black background",
        "on a reflective acrylic surface",
        "on a textured concrete surface",
        "on a pastel gradient backdrop",
        "on a wooden tabletop",
    ]
    lighting = [
        "studio softbox lighting",
        "hard rim lighting with crisp reflections",
        "high-key lighting",
        "low-key lighting with controlled highlights",
    ]
    camera = [
        "sharp edges",
        "high detail",
        "clean reflections",
        "product photography",
        "minimal composition",
        "macro detail shot",
    ]

    templates = [
        "Studio product photo of {prod} made of {mat}, {bg}, {light}, {cam}.",
        "Advertising photo of {prod} made of {mat}, {light}, {bg}, {cam}.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        prompt = _pick(rng, templates).format(
            prod=_pick(rng, products),
            mat=_pick(rng, materials),
            bg=_pick(rng, backgrounds),
            light=_pick(rng, lighting),
            cam=", ".join(rng.sample(camera, k=3)),
        )
        _unique_add(out, seen, prompt)
    return out


def gen_architecture(rng: random.Random, seen: set[str], count: int) -> list[str]:
    exteriors = [
        "a modern glass skyscraper",
        "a brutalist concrete museum",
        "a cozy brick townhouse",
        "a traditional wooden temple",
        "a futuristic curved bridge",
        "a minimalist concrete house",
        "an Art Deco theater facade",
        "a colorful Mediterranean street",
        "a mountain cabin with large windows",
        "a lighthouse on a cliff",
    ]
    interiors = [
        "a minimalist living room with large windows",
        "a cozy reading nook with warm lighting",
        "a modern kitchen with marble countertop",
        "a boutique hotel lobby with plants",
        "a quiet library with tall bookshelves",
        "a bright art studio with canvases",
        "a subway station platform with signage",
        "a clean laboratory workspace",
    ]
    lighting = [
        "golden hour light",
        "blue hour city lights",
        "soft overcast daylight",
        "bright midday sun",
        "interior warm lighting",
        "cool fluorescent lighting",
    ]
    camera = [
        "tilt-shift lens",
        "wide-angle lens",
        "straight vertical lines",
        "sharp architectural details",
        "high resolution",
        "realistic materials",
    ]
    weather = [
        "after rain with wet reflections",
        "on a clear day",
        "in light fog",
        "under dramatic clouds",
    ]
    templates = [
        "Architectural photo of {ext}, {weather}, {light}, {cam}.",
        "Wide-angle interior photo of {int}, {light}, {cam}, no people.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 250:
        attempts += 1
        if rng.random() < 0.5:
            prompt = _pick(rng, templates[:1]).format(
                ext=_pick(rng, exteriors),
                weather=_pick(rng, weather),
                light=_pick(rng, lighting),
                cam=", ".join(rng.sample(camera, k=3)),
            )
        else:
            prompt = _pick(rng, templates[1:]).format(
                int=_pick(rng, interiors),
                light=_pick(rng, lighting),
                cam=", ".join(rng.sample(camera, k=3)),
            )
        _unique_add(out, seen, prompt)
    return out


def gen_landscapes(rng: random.Random, seen: set[str], count: int) -> list[str]:
    scenes = [
        "a mountain range with a winding river",
        "a dense forest with sun rays through fog",
        "a desert landscape with sand dunes",
        "a tropical beach with turquoise water",
        "a snowy valley with pine trees",
        "a waterfall in a lush canyon",
        "a calm lake reflecting the sky",
        "a field of wildflowers under a big sky",
        "a rugged coastline with sea cliffs",
        "a volcanic landscape with black rocks",
    ]
    weather = [
        "misty morning",
        "sunset with warm colors",
        "sunrise with soft pastel sky",
        "after a rainstorm with clear air",
        "overcast moody light",
        "night sky with the Milky Way",
    ]
    camera = [
        "wide-angle landscape photography",
        "high detail",
        "sharp focus",
        "natural color grading",
        "dramatic clouds",
        "long exposure water",
        "aerial drone view",
    ]

    templates = [
        "Landscape photograph of {scene}, {weather}, {cam}.",
        "Cinematic landscape of {scene}, {weather}, {cam}.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        prompt = _pick(rng, templates).format(
            scene=_pick(rng, scenes),
            weather=_pick(rng, weather),
            cam=", ".join(rng.sample(camera, k=3)),
        )
        _unique_add(out, seen, prompt)
    return out


def gen_actions(rng: random.Random, seen: set[str], count: int) -> list[str]:
    subjects = [
        "two dancers",
        "a skateboarder",
        "a cyclist",
        "a chef",
        "a street musician",
        "a runner",
        "a barista",
        "a painter",
        "a climber",
        "a drummer",
    ]
    actions = [
        "mid-jump",
        "spinning",
        "pouring coffee in a thin stream",
        "sprinkling flour in the air",
        "playing an instrument with expressive motion",
        "running through shallow water",
        "painting a large canvas",
        "climbing a rock wall",
    ]
    settings = [
        "on a neon-lit city street at night",
        "in a sunlit studio",
        "in a crowded market",
        "on a rainy street with reflections",
        "in an industrial warehouse",
        "on a beach at sunset",
        "in a gym with dramatic lighting",
        "in a quiet cafe",
    ]
    camera = [
        "cinematic framing",
        "35mm wide-angle lens",
        "telephoto compression",
        "motion blur on background",
        "sharp subject",
        "high detail",
        "dynamic lighting",
    ]
    templates = [
        "Cinematic photo of {subj} {act} {setting}, {cam}.",
        "Action photograph of {subj} {act} {setting}, {cam}.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 250:
        attempts += 1
        prompt = _pick(rng, templates).format(
            subj=_pick(rng, subjects),
            act=_pick(rng, actions),
            setting=_pick(rng, settings),
            cam=", ".join(rng.sample(camera, k=3)),
        )
        _unique_add(out, seen, prompt)
    return out


def gen_counting_spatial(rng: random.Random, seen: set[str], count: int) -> list[str]:
    objects = [
        ("red apple", "red apples"),
        ("blue ceramic cup", "blue ceramic cups"),
        ("yellow rubber duck", "yellow rubber ducks"),
        ("green glass marble", "green glass marbles"),
        ("white candle", "white candles"),
        ("paper airplane", "paper airplanes"),
        ("silver coin", "silver coins"),
        ("chess pawn", "chess pawns"),
        ("strawberry", "strawberries"),
        ("macaron", "macarons"),
    ]
    surfaces = [
        "white plate",
        "black slate board",
        "wooden table",
        "blue fabric cloth",
        "marble countertop",
        "matte gray paper",
    ]
    patterns = [
        "in a straight line",
        "in a perfect circle",
        "in a 3 by 3 grid",
        "as a triangle",
        "as a spiral",
        "as two neat rows",
    ]
    lighting = [
        "soft daylight",
        "studio lighting",
        "overhead diffused light",
        "window light with gentle shadows",
    ]
    camera = [
        "top-down shot",
        "sharp focus",
        "high detail",
        "no extra objects",
        "clean background",
    ]
    relations = [
        ("a red cube", "a blue sphere", "to the left of"),
        ("a green cylinder", "a yellow cone", "to the right of"),
        ("a black mug", "a white plate", "in front of"),
        ("a silver key", "a brown wallet", "behind"),
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 400:
        attempts += 1
        if rng.random() < 0.65:
            singular, plural = rng.choice(objects)
            n = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
            prompt = (
                f"Top-down photo of exactly {n} {plural}, arranged { _pick(rng, patterns) } "
                f"on a { _pick(rng, surfaces) }, { _pick(rng, lighting) }, { ', '.join(rng.sample(camera, k=3)) }."
            )
        else:
            a, b, rel = rng.choice(relations)
            prompt = (
                f"A simple scene: {a} is {rel} {b} on a clean table, "
                f"{ _pick(rng, lighting) }, { ', '.join(rng.sample(camera, k=3)) }."
            )
        _unique_add(out, seen, prompt)
    return out


def gen_text_rendering(rng: random.Random, seen: set[str], count: int) -> list[str]:
    phrases = [
        "OPEN 24 HOURS",
        "FRESH BAKERY",
        "NO PARKING",
        "SILENT MODE",
        "SCIENCE FAIR 2026",
        "QUALITY FIRST",
        "HELLO WORLD",
        "LIMITED EDITION",
        "SAVE 50% TODAY",
        "PLEASE RECYCLE",
        "WELCOME HOME",
        "STAY CURIOUS",
        "KEEP GOING",
        "THANK YOU",
        "GOOD MORNING",
        "HARD WORK PAYS OFF",
        "DO NOT DISTURB",
        "NEW ARRIVALS",
        "URBAN GARDEN",
        "MOUNTAIN TRAIL",
        "欢迎光临",
        "请保持安静",
        "今日特价",
        "禁止停车",
        "学习使我快乐",
    ]
    media = [
        "a minimalist poster",
        "a street sign",
        "a book cover",
        "a product label",
        "a neon sign on a brick wall",
        "a clean UI screen",
        "a cafe menu board",
    ]
    typography = [
        "bold sans-serif typography",
        "clean serif typography",
        "monospace typography",
        "handwritten-style typography",
        "high-contrast typography",
    ]
    layout = [
        "centered text",
        "top-aligned header text with ample margins",
        "text inside a simple rounded rectangle",
        "text on a solid color background",
        "text with subtle drop shadow",
    ]
    colors = [
        "black text on white background",
        "white text on deep navy background",
        "yellow text on black background",
        "red text on cream background",
        "white neon text on dark background",
    ]
    constraints = [
        "the text is spelled exactly as given",
        "no extra words",
        "sharp readable letters",
        "clean edges",
        "high resolution",
    ]

    templates = [
        '{medium} with the exact text "{text}", {typo}, {layout}, {colors}, {cons}.',
        '{medium} showing the exact text "{text}", {typo}, {colors}, {layout}, {cons}.',
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 500:
        attempts += 1
        prompt = _pick(rng, templates).format(
            medium=_pick(rng, media).capitalize(),
            text=_pick(rng, phrases),
            typo=_pick(rng, typography),
            layout=_pick(rng, layout),
            colors=_pick(rng, colors),
            cons=", ".join(rng.sample(constraints, k=3)),
        )
        _unique_add(out, seen, prompt)
    return out


def gen_illustration_diagrams(rng: random.Random, seen: set[str], count: int) -> list[str]:
    styles = [
        "flat vector illustration",
        "isometric illustration",
        "technical line drawing",
        "blueprint-style diagram",
        "pixel art",
        "watercolor illustration",
        "oil painting",
        "3D render",
    ]
    concepts = [
        "a cross-section of a mountain showing geological layers",
        "a labeled diagram of a bicycle drivetrain",
        "an infographic of the water cycle with arrows",
        "a simple map of a fictional city with districts labeled",
        "a cutaway view of a modern apartment building",
        "a schematic of a coffee brewing setup",
        "a step-by-step illustration of folding a paper crane",
        "a minimal bar chart comparing three categories",
        "a minimalist UI icon set on a grid",
        "a flowchart describing a machine learning pipeline",
        "an exploded view of a mechanical watch",
        "a botanical illustration of a sunflower with labels",
    ]
    palettes = [
        "clean white background",
        "muted pastel palette",
        "limited two-color palette",
        "high-contrast black and white",
        "teal and orange palette",
    ]
    constraints = [
        "crisp lines",
        "readable labels",
        "high detail",
        "balanced composition",
        "minimal clutter",
    ]
    templates = [
        "{style} of {concept}, {palette}, {cons}.",
        "{style} depicting {concept}, {palette}, {cons}.",
    ]

    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 400:
        attempts += 1
        prompt = _pick(rng, templates).format(
            style=_pick(rng, styles),
            concept=_pick(rng, concepts),
            palette=_pick(rng, palettes),
            cons=", ".join(rng.sample(constraints, k=3)),
        )
        _unique_add(out, seen, prompt)
    return out


def generate_all(seed: int, total: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()

    generators = [
        ("portraits", gen_portraits),
        ("animals", gen_animals),
        ("food", gen_food),
        ("products", gen_products),
        ("architecture", gen_architecture),
        ("landscapes", gen_landscapes),
        ("actions", gen_actions),
        ("counting_spatial", gen_counting_spatial),
        ("text_rendering", gen_text_rendering),
        ("illustration_diagrams", gen_illustration_diagrams),
    ]

    per = total // len(generators)
    rem = total % len(generators)

    prompts: list[str] = []
    for idx, (_name, gen) in enumerate(generators):
        n = per + (1 if idx < rem else 0)
        prompts.extend(gen(rng, seen, n))

    # Final guard: if any generator hit attempt limit, backfill with safe mixed prompts.
    backfill_attempts = 0
    while len(prompts) < total and backfill_attempts < total * 200:
        backfill_attempts += 1
        fallback = _pick(
            rng,
            [
                "Photorealistic photo of a glass of water on a wooden table, soft daylight, sharp focus, high detail.",
                "Cinematic photo of a quiet street after rain, neon reflections, sharp focus, high detail.",
                "Studio product photo of a ceramic cup on a white background, softbox lighting, clean shadows, high detail.",
            ],
        )
        _unique_add(prompts, seen, fallback)

    return prompts[:total]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260108)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--num_calib", type=int, default=64)
    parser.add_argument("--out_eval", type=str, default="data/prompt_icml_comprehensive_1000.txt")
    parser.add_argument("--out_calib", type=str, default="data/prompt_icml_calib_64.txt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_path = (repo_root / args.out_eval).resolve()
    calib_path = (repo_root / args.out_calib).resolve()

    prompts = generate_all(seed=args.seed, total=args.num_eval)

    # Calibration prompts: deterministic sample across the eval set.
    rng = random.Random(args.seed + 1)
    if args.num_calib >= len(prompts):
        calib = prompts[:]
    else:
        calib = rng.sample(prompts, k=args.num_calib)

    _write_lines(eval_path, prompts)
    _write_lines(calib_path, calib)

    print(f"Wrote eval prompts:  {eval_path} ({len(prompts)})")
    print(f"Wrote calib prompts: {calib_path} ({len(calib)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

