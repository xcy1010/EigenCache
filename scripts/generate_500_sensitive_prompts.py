import os
import random

def main():
    # --- Basic Templates ---
    templates = [
        "A {texture} texture of {object}, {lighting} lighting, {style} style.",
        "A {structure} made of {material}, {lighting} lighting, {style} style.",
        "A {view} of {complex_scene}, {lighting} lighting, {style} style.",
        "A {text_scene} with the text '{text_content}' written in {font_style}, {lighting} lighting.",
        "A {interaction_scene}, {lighting} lighting, {style} style."
    ]

    # --- Sensitive Elements Library ---
    
    # 1. High-frequency textures (sensitive to blur)
    textures = [
        "dense wireframe mesh", "intricate lace", "carbon fiber weave", "snake skin scales", 
        "peacock feather", "butterfly wing scales", "fingerprint ridges", "circuit board traces",
        "knitted wool", "chainmail armor", "honeycomb pattern", "fractal spiral",
        "QR code pattern", "halftone dot pattern", "interference fringe", "diffraction grating"
    ]
    
    # 2. Complex objects (sensitive to structural deformation)
    objects = [
        "a mechanical watch movement", "a dandelion seed head", "a snowflake crystal", 
        "a spider web with dew drops", "a microchip under electron microscope", 
        "a human iris", "a dragonfly wing", "a cut diamond", "a fiber optic cable bundle",
        "a complex origami dragon", "a ship in a bottle", "a house of cards"
    ]
    
    # 3. Unstable structures (sensitive to positional drift)
    structures = [
        "precariously balanced tower of dice", "falling dominoes chain reaction", 
        "shattering glass sculpture", "splashing water crown", "exploding powder paint",
        "interlocking gears mechanism", "double helix DNA strand", "Escher-style impossible staircase",
        "Mandelbrot set zoom", "Fibonacci spiral of sunflowers"
    ]
    
    # 4. Materials (sensitive to lighting/caustics)
    materials = [
        "transparent glass", "polished chrome", "iridescent opal", "translucent jelly",
        "refractive diamond", "subsurface scattering wax", "holographic foil", "liquid mercury",
        "soap bubbles", "fiberglass", "aerogel", "obsidian"
    ]
    
    # 5. Complex scenes (sensitive to semantic consistency)
    complex_scenes = [
        "a busy Times Square crossing in rain", "a dense tropical rainforest floor", 
        "a library with thousands of books", "a crowd of people holding colorful umbrellas",
        "a factory assembly line with robot arms", "a coral reef with diverse fish",
        "a futuristic city skyline with flying cars", "a steampunk workshop with tools"
    ]
    
    # 6. Text scenes (sensitive to spelling)
    text_scenes = [
        "neon sign", "chalkboard menu", "printed receipt", "street graffiti", 
        "vintage typewriter page", "computer terminal screen", "newspaper headline",
        "book cover", "movie poster", "warning label"
    ]
    
    text_contents = [
        "HiCache", "ICML 2024", "Diffusion", "Generative AI", "Temporal Consistency",
        "Quantum Physics", "Lorem Ipsum", "The Quick Brown Fox", "Error 404", "System Failure"
    ]
    
    font_styles = [
        "glowing neon tubes", "calligraphy ink", "pixelated retro font", "3D bubble letters",
        "engraved metal", "embossed gold foil", "dripping blood paint", "matrix code rain"
    ]
    
    # 7. Interaction scenes (sensitive to attribute binding)
    interaction_scenes = [
        "a cat playing chess with a dog", "an astronaut riding a horse on Mars",
        "a robot painting a portrait of a human", "a teddy bear repairing a computer",
        "a skeleton playing a violin", "a fish swimming inside a lightbulb",
        "a tree growing out of a book", "a hand holding a burning flame"
    ]

    # 8. Style and Lighting
    lightings = [
        "cinematic", "volumetric", "rembrandt", "neon noir", "bioluminescent", 
        "harsh sunlight", "soft studio", "dramatic rim", "caustic", "strobe"
    ]
    
    styles = [
        "photorealistic", "cyberpunk", "steampunk", "macro photography", "electron microscope",
        "oil painting", "vector art", "blueprint", "thermal imaging", "x-ray"
    ]

    # --- Generation Logic ---
    prompts = set()
    target_count = 500
    
    while len(prompts) < target_count:
        template = random.choice(templates)
        
        if "{texture}" in template:
            p = template.format(
                texture=random.choice(textures),
                object=random.choice(objects),
                lighting=random.choice(lightings),
                style=random.choice(styles)
            )
        elif "{structure}" in template:
            p = template.format(
                structure=random.choice(structures),
                material=random.choice(materials),
                lighting=random.choice(lightings),
                style=random.choice(styles)
            )
        elif "{complex_scene}" in template:
            p = template.format(
                view=random.choice(["macro shot", "wide angle shot", "top-down view", "isometric view"]),
                complex_scene=random.choice(complex_scenes),
                lighting=random.choice(lightings),
                style=random.choice(styles)
            )
        elif "{text_scene}" in template:
            p = template.format(
                text_scene=random.choice(text_scenes),
                text_content=random.choice(text_contents),
                font_style=random.choice(font_styles),
                lighting=random.choice(lightings)
            )
        elif "{interaction_scene}" in template:
            p = template.format(
                interaction_scene=random.choice(interaction_scenes),
                lighting=random.choice(lightings),
                style=random.choice(styles)
            )
            
        prompts.add(p)

    # --- Write to file ---
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prompt_icml_sensitive_500.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for p in list(prompts):
            f.write(p + "\n")
            
    print(f"Successfully generated {len(prompts)} sensitive prompts at: {output_path}")

if __name__ == "__main__":
    main()
