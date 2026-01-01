"""
Braille Data Scraper - Collect native 8-dot braille from the internet

Sources of 8-dot braille content:
1. Unicode braille art (Twitter, Reddit, forums)
2. Braille music notation (8-dot standard)
3. Computer braille / Eurobraille (8-dot)
4. Accessibility documentation
5. GitHub repos with braille content
6. Academic papers on braille computing
7. Braille translation tools/APIs
"""

import re
import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

# Braille Unicode range
BRAILLE_PATTERN = re.compile(r'[\u2800-\u28FF]+')
DOT7_PATTERN = re.compile(r'[\u2840-\u287F]')  # Has dot 7
DOT8_PATTERN = re.compile(r'[\u2880-\u28BF]')  # Has dot 8
DOT78_PATTERN = re.compile(r'[\u28C0-\u28FF]')  # Has dots 7 and 8


@dataclass
class BrailleDocument:
    """A document containing braille content"""
    source: str
    url: str
    braille_text: str
    context: str  # Surrounding text for understanding
    has_dot7: bool
    has_dot8: bool
    timestamp: str
    byte_coverage: list  # Which byte values appear


def analyze_braille(text: str) -> dict:
    """Analyze braille content for 8-dot usage"""
    braille_chars = BRAILLE_PATTERN.findall(text)
    braille_str = ''.join(braille_chars)
    
    if not braille_str:
        return None
    
    # Check for dots 7 and 8
    has_dot7 = bool(DOT7_PATTERN.search(braille_str))
    has_dot8 = bool(DOT8_PATTERN.search(braille_str))
    has_dot78 = bool(DOT78_PATTERN.search(braille_str))
    
    # Calculate byte coverage
    byte_values = [ord(c) - 0x2800 for c in braille_str]
    unique_bytes = sorted(set(byte_values))
    
    return {
        "braille_text": braille_str,
        "length": len(braille_str),
        "has_dot7": has_dot7,
        "has_dot8": has_dot8,
        "has_dot78": has_dot78,
        "unique_patterns": len(unique_bytes),
        "byte_range": (min(unique_bytes), max(unique_bytes)) if unique_bytes else (0, 0),
        "is_8dot": has_dot7 or has_dot8,
    }


# ============================================================================
# Source-specific scrapers
# ============================================================================

async def scrape_github_braille():
    """Search GitHub for repositories with braille content"""
    
    # GitHub search queries for braille
    queries = [
        "braille unicode",
        "8-dot braille",
        "braille art",
        "braille music",
        "eurobraille",
        "computer braille",
        "⠀⠁⠂⠃",  # Search for actual braille
        "\\u2800",  # Unicode escape
    ]
    
    print("GitHub Braille Search Queries:")
    for q in queries:
        print(f"  https://github.com/search?q={q.replace(' ', '+')}&type=code")
    
    return queries


async def scrape_common_crawl():
    """
    Common Crawl has petabytes of web data.
    We can filter for pages containing braille Unicode.
    
    Strategy:
    1. Use Common Crawl Index to find URLs with braille
    2. Download those specific pages
    3. Extract braille + context
    """
    
    print("""
Common Crawl Strategy:
1. Query: https://index.commoncrawl.org/CC-MAIN-2024-10-index
2. Filter for pages containing U+2800-U+28FF
3. Download WARC records for matching URLs
4. Extract braille text with surrounding context

Example query:
  curl 'https://index.commoncrawl.org/CC-MAIN-2024-10-index?url=*.com&output=json' | 
  grep -E '[\u2800-\u28FF]'
""")


def generate_synthetic_8dot():
    """Generate synthetic 8-dot braille training data"""
    
    examples = []
    
    # All 256 byte-to-braille mappings
    for byte_val in range(256):
        braille_char = chr(0x2800 + byte_val)
        dots = []
        for i in range(8):
            if byte_val & (1 << i):
                dots.append(i + 1)
        
        examples.append({
            "instruction": f"Convert byte {byte_val} (0x{byte_val:02X}) to braille",
            "input": "",
            "output": f"Byte {byte_val} (0x{byte_val:02X}) = {braille_char} (dots: {','.join(map(str, dots)) or 'none'})"
        })
        
        examples.append({
            "instruction": f"What byte does {braille_char} represent?",
            "input": "",
            "output": f"{braille_char} represents byte {byte_val} (0x{byte_val:02X}), with dots {','.join(map(str, dots)) or 'none'} raised."
        })
    
    # Binary sequences
    test_sequences = [
        (b"Hello", "ASCII text"),
        (b"\x00\x01\x02\x03", "sequential bytes"),
        (b"\xFF\xFE\xFD\xFC", "high bytes"),
        (b"\x89PNG", "PNG header"),
        (b"PK\x03\x04", "ZIP header"),
        (b"\xFF\xD8\xFF\xE0", "JPEG header"),
        (b"%PDF", "PDF header"),
        (b"\x1F\x8B", "GZIP header"),
        (b"\x7FELF", "ELF executable"),
        (b"GIF89a", "GIF header"),
        (b"RIFF", "WAV/AVI header"),
        (b"\x00\x00\x01\x00", "ICO header"),
        (b"<!DOCTYPE", "HTML doctype"),
        (b'{"', "JSON start"),
    ]
    
    for data, desc in test_sequences:
        braille = ''.join(chr(0x2800 + b) for b in data)
        hex_str = data.hex().upper()
        
        examples.append({
            "instruction": f"Encode '{desc}' as braille",
            "input": "",
            "output": f"The {desc} bytes [{' '.join(f'0x{b:02X}' for b in data)}] encode to braille: {braille}"
        })
        
        examples.append({
            "instruction": f"What does this braille represent: {braille}",
            "input": "",
            "output": f"The braille {braille} decodes to bytes [{' '.join(f'0x{b:02X}' for b in data)}], which is a {desc}."
        })
    
    return examples


def find_braille_sources():
    """List known sources of 8-dot braille content"""
    
    sources = {
        "GitHub Repositories": [
            "https://github.com/wikipedia-braille - Wikipedia in braille",
            "https://github.com/search?q=braille+unicode - Code with braille",
            "https://github.com/search?q=braille+art - ASCII/braille art",
        ],
        "Academic": [
            "https://arxiv.org - Papers on braille computing",
            "https://scholar.google.com - Braille research",
            "https://www.researchgate.net - Braille accessibility papers",
        ],
        "Braille Organizations": [
            "https://www.brailleauthority.org - BANA standards",
            "https://www.eurobraille.fr - Eurobraille (8-dot)",
            "https://www.daisy.org - Digital accessible books",
            "https://www.bookshare.org - Accessible library",
        ],
        "Social Media": [
            "Twitter/X: Search for braille Unicode characters",
            "Reddit: r/braille, r/blind, r/accessibility",
            "Mastodon: Accessibility communities",
        ],
        "Braille Art": [
            "https://lachlanarthur.github.io/Braille-ASCII-Art/",
            "https://www.text-image.com/convert/braille.html",
            "Various braille art generators",
        ],
        "Music Braille": [
            "https://www.musicbraille.org - 8-dot music notation",
            "https://www.braillemusic.org - Music transcription",
        ],
        "Common Crawl": [
            "https://commoncrawl.org - Petabytes of web data",
            "Filter for pages with U+2800-U+28FF characters",
        ],
    }
    
    return sources


def main():
    print("=" * 60)
    print("8-Dot Braille Data Collection Strategy")
    print("=" * 60)
    
    # Show sources
    sources = find_braille_sources()
    print("\n📚 Known Sources of 8-Dot Braille:\n")
    for category, urls in sources.items():
        print(f"\n{category}:")
        for url in urls:
            print(f"  • {url}")
    
    # Generate synthetic data
    print("\n" + "=" * 60)
    print("Generating Synthetic 8-Dot Training Data...")
    print("=" * 60)
    
    examples = generate_synthetic_8dot()
    
    output_path = Path(__file__).parent / "synthetic_8dot.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated {len(examples)} synthetic examples")
    print(f"   Saved to: {output_path}")
    
    # Analyze coverage
    dot7_count = sum(1 for ex in examples if any(ord(c) >= 0x2840 for c in ex.get('output', '')))
    dot8_count = sum(1 for ex in examples if any(ord(c) >= 0x2880 for c in ex.get('output', '')))
    
    print(f"\n   Examples with dot 7: {dot7_count}")
    print(f"   Examples with dot 8: {dot8_count}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("""
1. Run GitHub scraper (requires API token):
   python braille_scraper.py --github

2. Query Common Crawl for braille pages:
   python braille_scraper.py --commoncrawl

3. Scrape braille art sites:
   python braille_scraper.py --art

4. Combine all data:
   python braille_scraper.py --combine
""")


if __name__ == "__main__":
    main()
