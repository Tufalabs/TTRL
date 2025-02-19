import json
import collections

def analyze_variants():
    with open('ttrl.json', 'r') as f:
        data = json.load(f)
        
    for question in data:
        # Count total variants
        total_variants = len(question['variants'])
        
        # Count variants by difficulty using Counter
        difficulty_counts = collections.Counter(
            variant['difficulty'] for variant in question['variants']
        )
        
        print(f"\nQuestion: {question['question']}")
        print(f"Total variants: {total_variants}")
        print("Variants by difficulty:")
        for difficulty, count in sorted(difficulty_counts.items()):
            print(f"  {difficulty}: {count}")

if __name__ == '__main__':
    analyze_variants()