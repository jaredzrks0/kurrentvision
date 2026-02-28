from pathlib import Path
from typing import Optional, List
from trdg.generators import GeneratorFromStrings # Locally imported
import os
import argparse

# IMPORTS REQUIRED FOR TRDG
import matplotlib
import tensorflow # Needed for handwriting


class KurrentTextGenerator:
    
    def __init__(self, font_path: str | None = None):
        if font_path is None:
            font_path = "data/Wiegel-Kurrent-Medium/WiegelKurrentMedium.ttf"
        
        self.font_path = Path(font_path)
        
        if not self.font_path.exists():
            raise FileNotFoundError(
                f"Font file not found at {self.font_path}. "
                f"Please check the font file location in data/Wiegel-Kurrent-Medium/"
            )
    
    def generate_images(
        self,
        texts: List[str],
        output_dir: str = "generated_text",
        count: int = 1,
        font_size: int = 32,
        skewing_angle: int = 0,
        random_skew: bool = False,
        blur: int = 0,
        random_blur: bool = False,
        background_type: int = 0,
        distorsion_type: int = 0,
        distorsion_orientation: int = 0,
        width: int = -1,
        alignment: int = 1,
        text_color: str = "#282828",
        orientation: int = 0,
        space_width: float = 1.0,
        character_spacing: int = 0,
        margins: tuple = (5, 5, 5, 5),
        fit: bool = False,
        handwritten: bool = False,
    ) -> List[Path]:
        """
        Generate synthetic text images.
        
        Args:
            texts: List of text strings to generate images from
            output_dir: Directory to save generated images
            count: Number of images to generate per text
            font_size: Size of the font
            skewing_angle: Angle of skewing (-25 to 25)
            random_skew: Apply random skewing
            blur: Blur intensity (0-3)
            random_blur: Apply random blur
            background_type: Type of background (0=gaussian noise, 1=plain white, 2=quasicrystal, 3=image)
            distorsion_type: Type of distortion (0=None, 1=Sine wave, 2=Cosine wave, 3=Random)
            distorsion_orientation: Orientation of distortion (0=Vertical, 1=Horizontal, 2=Both)
            width: Width of the generated image (-1 for automatic)
            alignment: Text alignment (0=left, 1=center, 2=right)
            text_color: Color of the text in hex format
            orientation: Text orientation (0=horizontal, 1=vertical)
            space_width: Width of spaces between words
            character_spacing: Spacing between characters
            margins: Margins (top, left, bottom, right)
            fit: Fit text to image size
            handwritten: Apply handwritten effect using TensorFlow model (requires TensorFlow installed)
                        EXPERIMENTAL - may produce more realistic handwriting appearance
            
        Returns:
            List of paths to generated images
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        images_dir = output_path / 'images'
        texts_dir = output_path / 'texts'
        images_dir.mkdir(parents=True, exist_ok=True)
        texts_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        global_idx = 0
        
        # Generate images for each text string
        for text in texts:
            generator = GeneratorFromStrings(
                strings=[text],
                count=count,
                fonts=[str(self.font_path)],
                language="de",
                size=font_size,
                skewing_angle=skewing_angle,
                random_skew=random_skew,
                blur=blur,
                random_blur=random_blur,
                background_type=background_type,
                distorsion_type=distorsion_type,
                distorsion_orientation=distorsion_orientation,
                width=width,
                alignment=alignment,
                text_color=text_color,
                orientation=orientation,
                space_width=space_width,
                character_spacing=character_spacing,
                margins=margins,
                fit=fit,
                is_handwritten=handwritten,
            )
            
            for idx, (img, lbl) in enumerate(generator):
                # Save the image
                filename = images_dir / f"text_{global_idx:05d}.png"
                img.save(filename)
                
                # Save the label
                label_filename = texts_dir / f"text_{global_idx:05d}.txt"
                with open(label_filename, 'w', encoding='utf-8') as f:
                    f.write(lbl)
                
                generated_files.append(filename)
                print(f"Generated: {filename} (Label: {lbl})")
                global_idx += 1
        
        return generated_files


def main():
    """Example usage of the KurrentTextGenerator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic text images with Kurrent font"
    )
    parser.add_argument(
        "--texts",
        type=str,
        help="Path to a text file containing one string per line",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_text",
        help="Output directory for generated images (default: generated_text)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of images to generate per text (default: 2)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=48,
        help="Font size (default: 48)",
    )
    
    args = parser.parse_args()
    
    # Load sample texts from file or use defaults
    if args.texts:
        text_file = Path(args.texts)
        if not text_file.exists():
            print(f"Error: Text file not found at {text_file}")
            return
        with open(text_file, 'r', encoding='utf-8') as f:
            sample_texts = [line.strip() for line in f if line.strip()]
        if not sample_texts:
            print(f"Error: No text found in {text_file}")
            return
    else:
        raise ValueError("Path to texts must be provided")

    try:
        generator = KurrentTextGenerator()
        
        print("Generating text images with Kurrent font...")
        generated_files = generator.generate_images(
            texts=sample_texts,
            output_dir=args.output,
            count=args.count,
            font_size=args.font_size,
            random_blur=True,
            random_skew=True,
            background_type=1,  # Plain white background
        )
        
        print(f"\nSuccessfully generated {len(generated_files)} images!")
        print(f"Images saved to: {args.output}/")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the Kurrent font file exists at:")
        print("data/Wiegel-Kurrent-Medium/WiegelKurrent-Medium.ttf")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
