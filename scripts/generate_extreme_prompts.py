import os

def main():
    prompts = [
        # --- Group 1: 密集文本与排版 (最容易拼错) ---
        "A neon sign board with the exact text: 'HiCache: High-Fidelity Cache for Diffusion Models' written in 5 lines, cyan and magenta colors, cyberpunk style.",
        "A handwritten letter on vintage paper, legible text reading: 'Dearest friend, the quick brown fox jumps over the lazy dog.', ink splatters, quill pen nearby.",
        "A complex restaurant menu board with prices and items: 'Burger $15, Fries $5, Shake $4', chalk on blackboard style, high detail.",
        "A futuristic HUD display showing code snippets: 'print(\"Hello World\")', 'if x > 0:', 'return True', green text on black background.",
        "A Scrabble board with tiles spelling out 'ARTIFICIAL INTELLIGENCE' horizontally and 'MACHINE LEARNING' vertically, wooden texture.",
        "A street graffiti wall with the word 'REVOLUTION' in 3D bubble letters, dripping paint, cracked brick texture behind.",
        "A typewriter typing the sentence: 'The future is now.', macro shot of the ink ribbon and metal keys.",

        # --- Group 2: 高频重复纹理 (最容易模糊/摩尔纹) ---
        "A close-up of a peacock feather, iridescent colors, thousands of tiny barbs clearly visible, macro photography.",
        "A basket of woven wicker texture, intricate interlocking patterns, sunlight casting complex shadows through the gaps.",
        "A dense field of lavender flowers, thousands of individual purple buds, bees hovering, shallow depth of field but sharp focus on flowers.",
        "A macro shot of a dragonfly's compound eye, thousands of hexagonal facets visible, metallic green color.",
        "A piece of knitted wool fabric, individual yarn fibers visible, soft lighting, cozy atmosphere.",
        "A close-up of a vinyl record, grooves clearly visible, dust particles, light reflecting off the black surface.",
        "A detailed topographic map with contour lines, tiny text labels for elevation, paper texture.",
        "A swarm of monarch butterflies filling the sky, thousands of orange and black wings, chaotic but detailed.",

        # --- Group 3: 复杂光影与折射 (最容易丢失高光) ---
        "A crystal glass of whiskey with ice cubes, complex light refraction through the ice and glass, golden liquid, cinematic lighting.",
        "A diamond necklace on black velvet, dispersion of white light into rainbow colors (fire), sparkling highlights.",
        "A soap bubble floating in the air, reflecting a distorted image of the surrounding room, thin film interference colors.",
        "A glass chess set on a mirror board, double reflections, caustic patterns on the table, ray tracing style.",
        "Raindrops on a window pane at night, city lights bokeh in the background, each droplet refracting the scene behind.",
        "A disco ball reflecting laser lights in a smoky room, thousands of light beams, high contrast.",
        "A bioluminescent jellyfish in deep ocean, translucent body, glowing tentacles, particle effects in water.",

        # --- Group 4: 复杂空间结构与几何 (最容易变形) ---
        "An impossible Escher-style staircase structure, infinite loops, architectural drawing style.",
        "A complex mechanical clockwork mechanism, gears interlocking, springs, brass and steel materials, steampunk style.",
        "A ship inside a glass bottle, detailed rigging, sails, and hull, glass distortion effects.",
        "A fractal Mandelbrot set, infinite self-similarity, colorful mathematical visualization.",
        "A DNA double helix structure, molecular model, atoms connected by bonds, scientific illustration style.",
        "A house of cards built 5 levels high, precarious balance, playing card textures visible.",
        "A detailed origami dragon made of patterned paper, sharp creases, paper texture."
    ]

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "prompt_icml_extreme_30.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
            
    print(f"Successfully generated 30 EXTREME prompts at: {output_path}")

if __name__ == "__main__":
    main()
